from fileinput import filename
from pyexpat.model import XML_CTYPE_MIXED
import xml.etree.ElementTree as ET
import os


def generate_urdf(path, mass=0.027, x_cm=0, y_cm=0, z_cm=0):

    tree = ET.parse(os.path.dirname(os.path.abspath(__file__))+"/cf2x_default.urdf")
    root = tree.getroot()

    # thrust to weight values    
    root[0].set('kf', '3.16e-10') # rpm to force coefficient
    root[0].set('km', '7.94e-12') # rpm to torque coefficient
    root[0].set('thrust2weight', '2.25') # Not need now

    # center of mass part
    original_mass = 0.027
    ixx = 1.4e-5 * mass / original_mass + mass * y_cm ** 2
    iyy = 1.4e-5 * mass / original_mass + mass * x_cm ** 2
    izz = ixx + iyy
    root.findall("link")[0].find('inertial').find('origin').set('rpy', '0 0 0')
    root.findall("link")[0].find('inertial').find('origin').set('xyz', '%s %s %s'%(str(x_cm), str(y_cm), str(z_cm)))
    root.findall("link")[0].find('inertial').find('mass').set('value', '%s'%str(mass))
    root.findall("link")[0].find('inertial').find('inertia').set('ixx', str(ixx))
    root.findall("link")[0].find('inertial').find('inertia').set('iyy', str(iyy))
    root.findall("link")[0].find('inertial').find('inertia').set('izz', str(izz))
    root.findall("link")[-1].find('inertial').find('origin').set('xyz', '%s %s %s'%(str(x_cm), str(y_cm), str(z_cm))) # center of mass -> Torque is applied based on CoM

    content = ET.tostring(root, encoding='unicode', method='xml')

    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),path)
    file_path = file_name.strip(os.path.basename(file_name))

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    with open(file_name, 'w') as f:
        f.seek(0,0)
        f.write("<?xml version=\"1.0\" ?>\n\n"+content)