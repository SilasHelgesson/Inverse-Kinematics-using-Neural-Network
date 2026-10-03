import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/glados/Desktop/Inverse-Kinematics-using-Neural-Network/install/target_generator'
