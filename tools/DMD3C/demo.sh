python demo.py gpus=[2] name=BP_KITTI ++chpt=checkpoints/dmd3c_kitti.pth \
    net=PMP num_workers=1 \
    data=KITTI data.testset.mode=test data.testset.height=352 \
    test_batch_size=1 metric=RMSE ++save=true
