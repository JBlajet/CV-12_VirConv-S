# torchrun --nproc_per_node=4 --master_port 4321 train_distill.py \
#     gpus=[0,1,2,3] num_workers=4 name=BP_KITTI \
#     net=PMP data=KITTI \
#     lr=1e-3 train_batch_size=2 test_batch_size=2 \
#     sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
#     nepoch=30 test_epoch=25 ++net.sbn=true

torchrun --nproc_per_node=1 --master_port 4321 train_distill.py \
    gpus=[0] num_workers=1 name=BP_KITTI \
    net=PMP data=KITTI \
    lr=1e-3 train_batch_size=2 test_batch_size=2 \
    sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
    nepoch=30 test_epoch=25 ++net.sbn=true