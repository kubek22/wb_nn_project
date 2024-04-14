import util
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./task_1_new_run')

kernels, feature_maps, input_img = util.visualize_kernels_and_feature_maps(308)
writer.add_images("conv1/2_kernels", kernels.unsqueeze(0), global_step=0)
writer.add_images("conv1/1_input_img", input_img, global_step=0)
for i, maps in enumerate(feature_maps):
    writer.add_images("conv1/3_feature_maps/{}".format(i), maps.unsqueeze(0), global_step=0)

writer.close()