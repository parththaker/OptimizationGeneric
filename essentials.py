from __future__ import print_function
import ConfigParser

def ReadConfig(filename):
    Config = ConfigParser.ConfigParser()
    Config.read(filename)
    return Config

def ConfigSectionMap(filename, section):
    dict1 = {}
    Config = ReadConfig(filename)
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1