'''
Client-side code that uses cbpy to configure and receive neural data from the
Blackrock Neural Signal Processor (NSP) (or nPlay).

from https://github.com/carmenalab/brain-python-interface/blob/master/riglib/blackrock/cerelink.py
'''

import sys
import time
from collections import namedtuple
from cerebus import cbpy

SpikeEventData = namedtuple("SpikeEventData",
                            ["chan", "unit", "ts", "arrival_ts"])
ContinuousData = namedtuple("ContinuousData",
                            ["chan", "samples", "arrival_ts"])

class BCIConnection(object):
    '''
    A wrapper around a UDP socket which sends the Blackrock NeuroPort system commands and
    receives data. Must run in a separte process (e.g., through `riglib.source`)
    if you want to use it as part of a task (e.g., BMI control)
    '''
    debug = False
    def __init__(self):
        self.parameters = dict()
        self.parameters['inst-addr']   = '192.168.137.128'
        self.parameters['inst-port']   = 51001
        self.parameters['client-port'] = 51002

        self.channel_offset = 0  # used to be 4 -- some old bug with nPlay
        print ('Using cbpy channel offset of:', self.channel_offset)

        if sys.platform == 'darwin':  # OS X
            print ('Using OS X settings for cbpy')
            self.parameters['client-addr'] = '255.255.255.255'
        else:  # linux
            print ('Using linux settings for cbpy')
            self.parameters['client-addr'] = '192.168.137.255'
            self.parameters['receive-buffer-size'] = 8388608

        self._init = False

        if self.debug:
            self.nsamp_recv = 0
            self.nsamp_last_print = 0

    def connect(self):
        '''Open the interface to the NSP (or nPlay).'''

        print ('calling cbpy.open in cerelink.connect()')
        # try:
        #     result, return_dict = cbpy.open(connection='default')
        #     time.sleep(3)
        # except:
        result, return_dict = cbpy.open(connection='default', parameter=self.parameters)
        time.sleep(3)

        print ('cbpy.open result:', result)
        print ('cbpy.open return_dict:', return_dict)
        # if return_dict['connection'] != 'Master':
        #     raise Exception
        print ('')

        # return_dict = cbpy.open('default', self.parameters)  # old cbpy

        self._init = True

    def select_channels(self, channels=None):
        '''Sets the channels on which to receive event/continuous data.

        Parameters
        ----------
        channels : array_like
            A sorted list of channels on which you want to receive data.
        '''

        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        buffer_parameter = {'absolute': True}  # want absolute timestamps
        """
        trial_config_params = {
            "buffer_parameter" : {
            "double"            : True,
            "continuous_length" : 35000,
            "event_length"      : 35000,
            "absolute"          : True,
            }
        }
        """

        # ability to select desired channels not yet implemented in cbpy
        # range_parameter = dict()
        # range_parameter['begin_channel'] = channels[0]
        # range_parameter['end_channel']   = channels[-1]

        print ('calling cbpy.trial_config in cerelink.select_channels()')
        result, reset = cbpy.trial_config(buffer_parameter=buffer_parameter)
        print ('cbpy.trial_config result:', result)
        print ('cbpy.trial_config reset:', reset)
        print ('')

    def start_data(self):
        '''Start the buffering of data.'''

        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        self.streaming = True

    def stop_data(self):
        '''Stop the buffering of data.'''

        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        print ('calling cbpy.trial_config in cerelink.stop()')
        result, reset = cbpy.trial_config(reset=False)
        print ('cbpy.trial_config result:', result)
        print ('cbpy.trial_config reset:', reset)
        print ('')

        self.streaming = False

    def disconnect(self):
        '''Close the interface to the NSP (or nPlay).'''

        if not self._init:
            raise ValueError("Please open the interface to Central/nPlay first.")

        print ('calling cbpy.close in cerelink.disconnect()')
        result = cbpy.close()
        print ('result:', result)
        print ('')

        self._init = False

    def __del__(self):
        self.disconnect()

    def get_event_data(self):
        '''A generator that yields spike event data.'''

        sleep_time = 0

        if self.streaming:
            obs = []

            result, trial = cbpy.trial_event(reset=True)  # TODO -- check if result = 0?
            arrival_ts = time.time()

            for list_ in trial:
                chan = list_[0]
                obs.append(list_[1])
            # time.sleep(sleep_time)
            return obs

    def get_trial_data(self):
        '''Get event and continuous data.'''

        sleep_time = 0
        print( "self.streaming: ", self.streaming)

        if self.streaming:
            obs_e = []
            obs_c = []
            # TODO: add timestamps
            result, events, cont = cbpy.trial_data(reset=True)  # TODO -- check if result = 0?
            arrival_ts = time.time()
            print ("result, events, cont: ", result, events, cont)

            for _e, _c in zip(events, cont):
                obs_e.append(_e[1]['events'])
                obs_c.append(_c[1])
            # time.sleep(sleep_time)
            return obs_c, obs_e

    def get_continuous_data(self):
        '''A generator that yields continuous data.'''
        
        data = cbpy.trial_continuous(reset=True)
        obs = data[1]
        ### TODO: Might still need to do some averaging across the samples here.
        return obs

