Main Dependencies
torch 1.1.0
torchvision 0.3.0
sklearn 0.20.3
numpy 1.19.5
matplotlib 3.0.3
CUDA 10.1

How to run:
You can use following command:
python svhnmain.py --batch_size=256 --n_epochs=500 --seed=17 --gamma_p=0.5 --gamma_c=0.2 --k2=5 --gpu=0 --dataset=SVHN

python mnistmain.py --batch_size=256 --n_epochs=500 --seed=17 --gamma_p=0.5 --gamma_c=0.2 --k2=5 --gpu=1 --dataset=MNIST

python fmnistmain.py --batch_size=256 --n_epochs=500 --seed=17 --gamma_p=0.5 --gamma_c=0.2 --k2=5 --gpu=2 --dataset=F-MNIST

option choice:

fmnistmain.py for F-MNIST dataset
svhnmain.py for SVHN dataset
mnistmain.py for MNIST dataset

--dataset =[SVHN,F-MNIST,MNIST]
--gamma_c can be any value in (0,1)
--gamma_p can be any value in (0,1)

k1 = [1,2,3,5] normal classes number
k2 = [1,2,3,5] pollution classes number