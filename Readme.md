# README

**English version below** 🇺🇸

---

## Deutsch 🇩🇪

### Projektbeschreibung

Dieses Projekt vergleicht zwei Ansätze zur Inversen Kinematik (IK) an einem 7-DoF Franka Emika Panda (FR3) Roboterarm:

1. **Neural Network-basierte IK** – Ein PyTorch-Modell, das offline auf MoveIt-generierten Labels trainiert wird und online mit dem Roboter verfeinert wird
2. **Traditionelle analytische IK** – MoveIt-Service mit dem Standard-KDL-Solver

Das Projekt misst und vergleicht Genauigkeit, Geschwindigkeit und Zuverlässigkeit beider Methoden in einem ROS 2-Ecosystem.

### Architektur

Das Projekt besteht aus drei ROS 2-Paketen:

```
src/
├── neural_predictor/     NN-basierte IK mit Online-Training
├── traditional_predictor/ MoveIt-basierter analytischer Solver
└── target_generator/     Zufällige Zielpose-Generierung für Experimente
```

**Datenfluss:**
```
target_generator → Random poses (geometry_msgs/PoseStamped)
                ↓
        ┌───────┴────────┐
        ↓                ↓
neural_predictor    traditional_predictor
        ↓                ↓
        └───────┬────────┘
                ↓
        Joint states (sensor_msgs/JointState)
        → Metrics, Error Logging
```

### Komponenten

#### neural_predictor
- `IKNetwork`: Ein einfaches 3-schichtiges Feedforward-Netzwerk (3→128→64→7)
- **Online-Training**: Lädt vorgenerierte Labels auf Anfrage, verfeinert die Gewichte mit Gradient Descent
- **Inference**: Schnelle GPU-beschleunigte Vorhersagen (typisch < 1 ms)
- Eingabe: Zielposition (x, y, z)
- Ausgabe: 7 Gelenkwinkel (rad)

#### traditional_predictor
- Wrapper um MoveIt's `GetPositionIK`-Service
- Analytische Lösung ohne Lernen
- Höhere Rechenzeit, aber potenziell robuster für neue Szenarien
- Benötigt einen laufenden MoveIt-Manager und Roboter-URDF

#### target_generator
- Erzeugt Zufallsziele innerhalb des erreichbaren Arbeitsraums des FR3
- Veröffentlicht `PoseStamped` Nachrichten mit konfigurierbarer Frequenz

### Voraussetzungen

- Ubuntu 22.04 oder 24.04 (ROS 2 Jazzy oder Humble)
- ROS 2 installiert (https://docs.ros.org/en/jazzy/Installation.html)
- Python 3.10+
- MoveIt 2 (`sudo apt install ros-<distro>-moveit2`)
- PyTorch (`pip install torch`)
- **Franka Workspace**: Separater Workspace mit `franka_bringup`, `franka_moveit_config` und `franka_description`
  - Muss installiert und gebaut sein
  - Muss VOR diesem Projekt ge-sourcet werden

### Installation

1. **Franka Workspace vorbereiten**
```bash
# Franka Workspace sourcing (anpassen je nach Pfad)
source ~/franka_ws/install/setup.bash
```

2. **Repository klonen**
```bash
git clone https://github.com/SilasHelgesson/Inverse-Kinematics-using-Neural-Network
cd Inverse-Kinematics-using-Neural-Network
```

3. **Workspace aufbauen**
```bash
colcon build --symlink-install
source install/setup.bash
```

4. **Abhängigkeiten installieren**
```bash
pip install torch numpy
```

**Wichtig**: Der Franka Workspace muss VOR dem sourcing dieses Projektes ge-sourcet werden, damit die ROS 2-Umgebung alle Franka-Pakete findet.

### Ausführung

#### Single Launch File (empfohlen)
Das gesamte System wird mit einer einzigen Launch-Datei gestartet, die alle Komponenten orchestriert:

**Terminal 1: Franka Workspace + MoveIt mit Fake Hardware**
```bash
source ~/franka_ws/install/setup.bash
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py use_fake_hardware:=true robot_ip:=127.0.0.1
```

**Terminal 2: Franka Workspace + IK Vergleich starten**
```bash
source ~/franka_ws/install/setup.bash
source install/setup.bash
ros2 launch neural_predictor main_ik_comparison.launch.py
```

Die `main_ik_comparison.launch.py` startet automatisch:
- Traditional IK Solver (MoveIt-basiert)
- Neural Network IK Solver (mit Online-Training)
- Robot State Publisher für NN-Visualisierung
- Random Target Generator
- Marker Publisher für RViz-Visualisierung

RViz sollte automatisch öffnen und beide IK-Lösungen side-by-side anzeigen.

### Konfiguration

Die Parameter befinden sich in den `launch/`-Dateien:

- `neural_predictor/launch/nn_robot.launch.py`
  - `model_path`: Pfad zu `ik_model.pth`
  - `learning_rate`: Lernrate für Online-Training
  - `batch_size`: Minibatch-Größe beim Training
  - `training_samples`: Anzahl Trainings-Iterationen pro Ziel

- `target_generator/launch/target_generator.launch.py`
  - `target_rate`: Hz für Ziel-Veröffentlichung
  - `workspace_bounds`: Min/Max xyz-Grenzen

### Trainierte Modelle

Das Projekt beinhaltet ein vortrainiertes Modell:
- **`ik_model.pth`** – Auf 10.000 zufälligen FR3-Zielkonfigurationen trainiert

Neutraining (optional):
```python
# In neural_predictor/neural_predictor/online_training_ik.py
# Modell wird online bei Bedarf neu trainiert
```

### Projektstruktur

```
src/
  neural_predictor/
    ├── neural_predictor/
    │   └── online_training_ik.py    # Hauptmodul
    ├── launch/
    │   ├── nn_robot.launch.py
    │   └── main_ik_comparison.launch.py
    └── package.xml

  traditional_predictor/
    ├── traditional_predictor/
    │   └── traditional_ik_solver.py # MoveIt-Wrapper
    └── package.xml

  target_generator/
    ├── target_generator/
    │   ├── marker_publisher.py
    │   └── random_target_node.py
    └── package.xml

ik_model.pth                          # Trainiertes PyTorch-Modell
```

### Lizenz

Dieses Projekt ist unter der [GNU General Public License v3.0 oder später](LICENSE) lizenziert.

---

## English 🇺🇸

### Project Description

This project compares two approaches to inverse kinematics (IK) on a 7-DoF Franka Emika Panda (FR3) robot arm:

1. **Neural Network-based IK** – A PyTorch model trained offline on MoveIt-generated labels and refined online with the robot
2. **Traditional Analytical IK** – MoveIt service with the standard KDL solver

The project measures and compares accuracy, speed, and reliability of both methods in a ROS 2 ecosystem.

### Architecture

The project consists of three ROS 2 packages:

```
src/
├── neural_predictor/     NN-based IK with online training
├── traditional_predictor/ MoveIt-based analytical solver
└── target_generator/     Random target pose generation for experiments
```

**Data Flow:**
```
target_generator → Random poses (geometry_msgs/PoseStamped)
                ↓
        ┌───────┴────────┐
        ↓                ↓
neural_predictor    traditional_predictor
        ↓                ↓
        └───────┬────────┘
                ↓
        Joint states (sensor_msgs/JointState)
        → Metrics, Error Logging
```

### Components

#### neural_predictor
- `IKNetwork`: A simple 3-layer feedforward network (3→128→64→7)
- **Online Training**: Loads pre-generated labels on demand, refines weights with gradient descent
- **Inference**: Fast GPU-accelerated predictions (typical < 1 ms)
- Input: Target position (x, y, z)
- Output: 7 joint angles (rad)

#### traditional_predictor
- Wrapper around MoveIt's `GetPositionIK` service
- Analytical solution without learning
- Higher computation time but potentially more robust for novel scenarios
- Requires a running MoveIt manager and robot URDF

#### target_generator
- Generates random targets within FR3's reachable workspace
- Publishes `PoseStamped` messages at configurable frequency

### Prerequisites

- Ubuntu 22.04 or 24.04 (ROS 2 Jazzy or Humble)
- ROS 2 installed (https://docs.ros.org/en/jazzy/Installation.html)
- Python 3.10+
- MoveIt 2 (`sudo apt install ros-<distro>-moveit2`)
- PyTorch (`pip install torch`)
- **Franka Workspace**: Separate workspace with `franka_bringup`, `franka_moveit_config`, and `franka_description`
  - Must be built and installed
  - Must be sourced BEFORE this project

### Installation

1. **Prepare Franka Workspace**
```bash
# Source Franka workspace first (adjust path as needed)
source ~/franka_ws/install/setup.bash
```

2. **Clone the repository**
```bash
git clone https://github.com/SilasHelgesson/Inverse-Kinematics-using-Neural-Network
cd Inverse-Kinematics-using-Neural-Network
```

3. **Build the workspace**
```bash
colcon build --symlink-install
source install/setup.bash
```

4. **Install dependencies**
```bash
pip install torch numpy
```

**Important**: The Franka workspace must be sourced BEFORE sourcing this project, so that the ROS 2 environment can find all Franka packages.

### Execution

#### Single Launch File (Recommended)
The entire system is started with a single launch file that orchestrates all components:

**Terminal 1: Source Franka Workspace + MoveIt with Fake Hardware**
```bash
source ~/franka_ws/install/setup.bash
source install/setup.bash
ros2 launch franka_fr3_moveit_config moveit.launch.py use_fake_hardware:=true robot_ip:=127.0.0.1
```

**Terminal 2: Source Franka Workspace + Start IK Comparison**
```bash
source ~/franka_ws/install/setup.bash
source install/setup.bash
ros2 launch neural_predictor main_ik_comparison.launch.py
```

The `main_ik_comparison.launch.py` automatically starts:
- Traditional IK Solver (MoveIt-based)
- Neural Network IK Solver (with online training)
- Robot State Publisher for NN visualization
- Random Target Generator
- Marker Publisher for RViz visualization

RViz should open automatically and display both IK solutions side-by-side.

### Configuration

Parameters are located in the `launch/` files:

- `neural_predictor/launch/nn_robot.launch.py`
  - `model_path`: Path to `ik_model.pth`
  - `learning_rate`: Learning rate for online training
  - `batch_size`: Minibatch size during training
  - `training_samples`: Number of training iterations per target

- `target_generator/launch/target_generator.launch.py`
  - `target_rate`: Hz for target publishing
  - `workspace_bounds`: Min/max xyz bounds

### Pre-trained Models

The project includes a pre-trained model:
- **`ik_model.pth`** – Trained on 10,000 random FR3 target configurations

Retraining (optional):
```python
# In neural_predictor/neural_predictor/online_training_ik.py
# Model is refined online on demand
```

### Project Structure

```
src/
  neural_predictor/
    ├── neural_predictor/
    │   └── online_training_ik.py    # Main module
    ├── launch/
    │   ├── nn_robot.launch.py
    │   └── main_ik_comparison.launch.py
    └── package.xml

  traditional_predictor/
    ├── traditional_predictor/
    │   └── traditional_ik_solver.py # MoveIt wrapper
    └── package.xml

  target_generator/
    ├── target_generator/
    │   ├── marker_publisher.py
    │   └── random_target_node.py
    └── package.xml

ik_model.pth                          # Trained PyTorch model
```

### License

This project is licensed under the [GNU General Public License v3.0 or later](LICENSE).