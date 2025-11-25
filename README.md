# LSPE

This repository contains the implementation of **Latent State-Predictive Exploration (LSPE)**. It supports both simulation training in Habitat and real-world deployment on Unitree Go2 robots via ROS2.

---

## Simulation Environment Setup (Habitat)

To run LSPE in the simulation environment, you need to set up the Habitat simulator and dependencies.

### Installing Dependencies

Follow these steps to set up the environment:

1. **Clone the repository and install basic requirements**
    ```bash
   git clone --recurse-submodules [LSPE_REPO_URL]
   cd LSPE
   pip install -e .
   pip install -r requirements.txt
    ```

2. **Create a new conda environment**

    ```bash
    conda create -n lspe python=3.9 cmake=3.14.0
    conda activate lspe
    conda install habitat-sim withbullet -c conda-forge -c aihabitat
    ```

3.  **Clone and install `habitat-lab` and `habitat-baselines`**

    ```bash
    cd src
    git clone --branch stable [https://github.com/facebookresearch/habitat-lab.git](https://github.com/facebookresearch/habitat-lab.git)
    cd habitat-lab
    pip install -e .
    pip install -e habitat-baselines
    ```

4.  **Install additional dependencies**

    ```bash
    conda install git git-lfs
    ```

5.  **Download the required datasets**

    ```bash
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
    python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
    # Note: HM3D dataset requires official authentication
    python -m habitat_sim.utils.datasets_download --username <your_username> --password <your_password> --uids hm3d_minival_v0.2
    ```

6.  **Verify the installation**
    Run an example to ensure everything is set up correctly:

    ```bash
    python src/habitat-lab/examples/example.py
    ```
### Test Experiments

After successfully installing `habitat-lab` and `habitat-sim` dependencies, you can run a test experiment to ensure everything is set up correctly.

#### Running the Training Script

Navigate to the `habitat-lab` directory and execute the `train_lspe.py` script:

```bash
python train_lspe.py
```
-----

## Real Robot Deployment (ROS2 & Hardware)

### Description:
This is a code file based on ROS2. It utilizes the high-level control of topics in the Unitree Go2. It extracts camera images from the Unitree Go2, feeds them as input to the model, and the model outputs actions. These actions are then converted into ROS2 messages and sent to the control-related Unitree Go2 topics to drive the real robot's actions, thereby completing the overall task.

### Relevant documents:

In order to build the environment, you may need:

#### ROS2:

ROS2 https://docs.ros.org

Humble https://docs.ros.org/en/humble/

Foxy https://docs.ros.org/en/foxy/index.html#ros-2-documentation

#### Unitree：

Unitree Go2 https://support.unitree.com/home/en/developer/about_Go2

Unitree Go2 SDK https://github.com/unitreerobotics/unitree_sdk2

Unitree Go2 ROS2 https://github.com/unitreerobotics/unitree_ros2

### Package Structure:
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

A good way to solve it, is downloading the unitree msgs ros2 pkg directly, named __unitree_go__ .

You can find it in __Relevant documents__ .

Then, add it to your work space.

For example:
```
Your_Own_WorkSpace/
└── src/
    └── real_robot_pkg/
    │   ├── real_robot_pkg/
    │   │   ├── __init__.py
    │   │   └── policy_node.py
    │   ├── resource/
    │   │   └── real_robot_pkg
    │   ├── test/
    │   │   ├── test_copyright.py
    │   │   ├── test_flake8.py
    │   │   └── test_pep257.py
    │   ├── package.xml
    │   ├── setup.cfg
    │   └── setup.py
    └── unitree_go/
```

### Usage:
Open your own workspace:
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
