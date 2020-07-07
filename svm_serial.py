#!/usr/bin/env python3
#from __future__ import absolute_import, division, print_function
#import importlib
import threading
import serial
import time

import numpy as np
import joblib

from pprzlink.message import PprzMessage
from pprzlink.pprz_transport import PprzTransport


class SerialMessagesInterface(threading.Thread):
    def __init__(self, callback, verbose=False, device='/dev/ttyUSB0', baudrate=115200,
                 msg_class='telemetry', interface_id=0):
        threading.Thread.__init__(self)
        self.callback = callback
        self.verbose = verbose
        self.msg_class = msg_class
        self.id = interface_id
        self.running = True
        try:
            self.ser = serial.Serial(device, baudrate, timeout=1.0)
        except serial.SerialException:
            print("Error: unable to open serial port '%s'" % device)
            exit(0)
        self.trans = PprzTransport(msg_class)

    def stop(self):
        print("End thread and close serial link")
        self.running = False
        self.ser.close()

    def shutdown(self):
        self.stop()

    def __del__(self):
        try:
            self.ser.close()
        except:
            pass

    def send(self, msg, sender_id,receiver_id = 0, component_id = 0):
        """ Send a message over a serial link"""
        if isinstance(msg, PprzMessage):
            data = self.trans.pack_pprz_msg(sender_id, msg, receiver_id, component_id)
            self.ser.write(data)
            self.ser.flush()

    def run(self):
        """Thread running function"""
        try:
            while self.running:
                # Parse incoming data
                c = self.ser.read(1)
                if len(c) == 1:
                    if self.trans.parse_byte(c):
                        (sender_id, receiver_id, component_id, msg) = self.trans.unpack()
                        #self.collect_data(msg)
                        if self.verbose:
                            print("New incoming message '%s' from %i (%i) to %i" % (msg.name, sender_id, component_id, receiver_id))
                        # Callback function on new message
                        if self.id == receiver_id:
                            self.callback(sender_id, msg)

        except StopIteration:
            pass

class Model():
    def __init__(self, saved_model_filename, saved_scaler_filename, verbose=False):
        self.model = joblib.load(saved_model_filename) # model can be saved with joblib.dump(clf,'fname.joblib')
        self.scaler = joblib.load(saved_scaler_filename)
        self.interface = None
        self.logger = None
        self.verbose = verbose
        self.fault_info  = 0
        self.fault_type = 0
        self.fault_class = 0
        self.fault_state_increment = 0

    def set_interface(self,interface):
        self.interface = interface

    def set_logger(self, logger):
        self.logger = logger

    def predict(self,X):
        self.logger.write_log(X)
        X_scaled = self.scaler.transform(X.reshape(1,-1))
        #self.logger.write_log(X_scaled)
        #print('X_scaled shpae : ',X_scaled.shape)
        self.fault_info = self.model.predict(X_scaled)
        self.update_fault_state()

    def update_fault_state(self):
        if self.fault_type != self.fault_info :
            self.fault_state_increment += 1
        else:
            self.fault_state_increment = 0
        if self.fault_state_increment >= 3 : self.fault_type = self.fault_info
        self.send_pprz_fault_info()

    def send_pprz_fault_info(self):
        if self.verbose: print('Sending the FAULT_INFO')
        set_fault = PprzMessage('datalink', 'FAULT_INFO')
        set_fault['info'] = self.fault_info
        set_fault['type'] = self.fault_type
        set_fault['class']= self.fault_class
        self.interface.send(set_fault, 0)

class Logger():
    def __init__(self,filename='log.txt'):
        self.filename = filename
        self.log = np.zeros((8000, 161))
        self.line = 0

    def write_log(self, X):
        self.log[self.line,0] = time.time()
        self.log[self.line,1:]=X
        self.line += 1

    def close_log(self):
        with open(self.filename, 'wb') as f:
            np.save(f,self.log[:self.line])


class Data_Collector():
    def __init__(self, model=None, dimension=8, history_step=2, verbose=False):
        self.verbose = verbose
        self.model = model
        self.history_step = history_step
        self.dimension=dimension
        self.data = np.zeros([self.dimension*self.history_step])
        self.X = np.zeros([self.dimension])
        self.msg_received = np.zeros([3])
        self.accel_msg_length = 3
        self.gyro_msg_length = 3
        self.commands_msg_length = 2
        #self.last_time=time.time()
        #self.last_gyro_time = time.time()

    def set_model(self, model):
        self.model = model

    def parse_msg(self, sender_id, msg):
        if msg.name == 'IMU_GYRO':
            #print('Gyro time : ',time.time()-self.last_gyro_time)
            #self.last_gyro_time = time.time()
            if self.verbose: print('GYRO received',msg.get_field(0), msg.get_field(1), msg.get_field(2) )
            self.set_gyro_data(msg)
        if msg.name == 'IMU_ACCEL':
            if self.verbose: print('ACCEL received',msg.get_field(0), msg.get_field(1), msg.get_field(2) )
            self.set_accel_data(msg)
        if msg.name == 'COMMANDS':
            if self.verbose: print('COMMANDS received',msg._fieldvalues[0] )
            self.set_commands_data(msg)

    def set_accel_data(self,msg):
        n = int(self.msg_received[0]*self.dimension)
        nn = int(n + self.accel_msg_length)
        #print('accel:',n,nn)
        self.data[n:nn] = msg.get_field(0), msg.get_field(1), msg.get_field(2) 
        self.msg_received[0] += 1
        self.check_data_ready()

    def set_gyro_data(self,msg):
        n = int(self.msg_received[1]*self.dimension + self.accel_msg_length)
        nn = int(n + self.gyro_msg_length)
        #print('gyro:',n,nn)
        #if msg.get_field(0) == 0: print('Zero Value in Gyro X')
        #if msg.get_field(1) == 0: print('Zero Value in Gyro Y')
        #if msg.get_field(2) == 0: print('Zero Value in Gyro Z')
        self.data[n:nn] = msg.get_field(0), msg.get_field(1), msg.get_field(2) 
        self.msg_received[1] += 1
        self.check_data_ready()

    def set_commands_data(self,msg):
        n = int(self.msg_received[2]*self.dimension + self.accel_msg_length + self.gyro_msg_length)
        nn = int(n + self.commands_msg_length)
        #print('commands :',n,nn)
        #print('Commands converted : ', msg._fieldvalues[0][1:3])
        self.data[n:nn] = msg._fieldvalues[0][1:3] # get only the right and left ailevon commands
        self.msg_received[2] += 1
        self.check_data_ready()

    def reset_msg(self):
        self.msg_received[:] = 0

    def send_data_to_model(self,X):
        if self.model != None :
            #print('Prediction time : ',time.time()-self.last_time)
            #self.last_time = time.time()
            self.model.predict(X)
            #pass
        else:
            print('No prediction model assigned !')
            pass

    def check_data_ready(self):
        if (self.msg_received[0]>=self.history_step) & (self.msg_received[1]>=self.history_step) & (self.msg_received[2]>=self.history_step):
            self.X = self.data.copy()
            self.send_data_to_model(self.X)
            self.reset_msg()
            if self.verbose: print('X :',self.X)
            


def main():
    '''
    Real-time Fault prediction module:

    '''
    import time
    import argparse
    from pprzlink import messages_xml_map
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="path to messages.xml file", default='pprzlink/messages.xml')
    parser.add_argument("-c", "--class", help="message class", dest='msg_class', default='telemetry')
    parser.add_argument("-d", "--device", help="device name", dest='dev', default='/dev/serial0')
    parser.add_argument("-b", "--baudrate", help="baudrate", dest='baud', default=230400, type=int)
    parser.add_argument("-id", "--ac_id", help="aircraft id (receiver)", dest='ac_id', default=42, type=int)
    parser.add_argument("--interface_id", help="interface id (sender)", dest='id', default=0, type=int)
    args = parser.parse_args()
    messages_xml_map.parse_messages(args.file)

    #model_filename = 'svm_model_1.joblib'
    #scaler_filename = 'svm_scaler_1.joblib'

    #model_filename = 'models/svm_model_binary_r05.joblib'
    #scaler_filename = 'models/svm_scaler_binary_r05.joblib'

    #model_filename = 'models/svm_model_binary_r05_07072020_AGC_20h_10s.joblib'
    #scaler_filename = 'models/svm_scaler_binary_r05_07072020_AGC_20h_10s.joblib'

    model_filename = 'models/svm_model_binary_r03_10072020_AGC_20h_10s.joblib'
    scaler_filename = 'models/svm_scaler_binary_r03_10072020_AGC_20h_10s.joblib'

    model = Model(model_filename, scaler_filename)
    #collector = Data_Collector(model=model, dimension=8, history_step=20, verbose=True)
    collector = Data_Collector(model=model, dimension=8, history_step=20, verbose=False)
    serial_interface = SerialMessagesInterface(collector.parse_msg, device=args.dev,
                                               baudrate=args.baud, msg_class=args.msg_class, interface_id=args.id, verbose=False)
    logger = Logger()
    model.set_interface(serial_interface)
    model.set_logger(logger)

    print("Starting serial interface on %s at %i baud" % (args.dev, args.baud))
    try:
        serial_interface.start()

        # give the thread some time to properly start
        time.sleep(0.1)

        while serial_interface.isAlive():
            serial_interface.join(1)
    except (KeyboardInterrupt, SystemExit):
        print('Shutting down...')
        logger.close_log()
        serial_interface.stop()
        exit()


if __name__ == '__main__':
    main()
