# learning_triple_poses

Based on CVPR2022 best paper **Learning to Solve Hard Minimal Problems** implements solving the relative position of three images.

## 1. Build

### 1.1 dependencies
- Eigen
- OpenCV

### 1.2 build

clone this project 

```
git clone https://github.com/linyicheng1/leanring_triple_poses.git
```

build project 

```
mkdir build & cd build 
```

```
cmake .. & make -j
```

## 2. Usage

### 2.1 Extract feature points

```
./build/sift <img1_dir> <img2_dir> <img3_dir> <output.txt>
```

For example:

```
./build/sift data/000570.png data/000575.png data/000580.png data/matches.txt
```

After the cmd you will see the match results

![$_(OF(@ZD7H%KLC$XUN~N4X](https://github.com/linyicheng1/leanring_triple_poses/assets/50650063/bf373e21-3993-47a4-b94c-28ead4bb4554)

### 2.2 Solve poses

```
./build/learning_triple_poses <model_path> <set_path> <data_path>
```

For example:

```
./build/learning_triple_poses ./model ./model/trainParam.txt data/matches.txt
```

After the cmd you will see some poses output, and the groundtruth is:

```
t01: 
-0.0116375
-0.022658
 0.999676
t02: 
-0.0224208
-0.0452998
 1.99962
```

You can find the correct result in the output bit position, if not you need to run the programme again.

