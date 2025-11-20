# LSPE

## Relevant documents:

In order to build the environment, you may need:

### ROS2:

ROS2 https://docs.ros.org

Humble https://docs.ros.org/en/humble/

Foxy https://docs.ros.org/en/foxy/index.html#ros-2-documentation

### Unitree：

Unitree Go2 https://support.unitree.com/home/en/developer/about_Go2

Unitree Go2 SDK https://github.com/unitreerobotics/unitree_sdk2

Unitree Go2 ROS2 https://github.com/unitreerobotics/unitree_ros2

## Package Structure:
- git clone this repo
- move it to your workspace
- make sure the structure is correct

If Unitree Go2 environment is correctly configured, the structure will be like:
For example:
```
Your_Own_WorkSpace/
└── src/
    └── real_robot_pkg/
        ├── real_robot_pkg/
        │   ├── __init__.py
        │   └── policy_node.py
        ├── resource/
        │   └── real_robot_pkg
        ├── test/
        │   ├── test_copyright.py
        │   ├── test_flake8.py
        │   └── test_pep257.py
        ├── package.xml
        ├── setup.cfg
        └── setup.py
```

If unfortunately the Unitree Go2 environment configuration fails.
A good way to solve it, is downloading the unitree msgs ros2 pkg directly.
You can find it in __Relevant documents__
Then, add it to your work space.
For example:
```
Your_Own_WorkSpace/
└── src/
    └── real_robot_pkg/
        ├── real_robot_pkg/
        │   ├── __init__.py
        │   └── policy_node.py
        ├── resource/
        │   └── real_robot_pkg
        ├── test/
        │   ├── test_copyright.py
        │   ├── test_flake8.py
        │   └── test_pep257.py
        ├── package.xml
        ├── setup.cfg
        └── setup.py
```

## Usage:
Open your own work space:
```
cd Your_Own_WorkSpace/
```
Build ROS2 Package:
```
colcon build
```
Source:
```
source install/setup.bash
```
Run:
```
ros2 run real_robot_pkg policy_node
```
