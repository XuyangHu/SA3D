import tensorflow as tf
import os
import argparse
from utils import *
from loss import vae_criterion
from vae_unet import VAE_UNet

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of VAE_UNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--data_path', type=str, default='./dataset/train/', help='dataset_name')
    parser.add_argument('--mode', type=str, default='front_to_leftside', help='mode of image translation')
    parser.add_argument('--test_data_path', type=str, default='./test_data', help='test dataset_name')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')
    parser.add_argument('--gf_dim', type=int, default=64, help='The number of channel')
    parser.add_argument('--out_dim', type=int, default=3, help='The number of categories')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--max_iter', type=int, default=50000, help='The number of iters')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue train')
    parser.add_argument('--pretrained_model_path', type=str, default='./checkpoint/', help='pretrained model path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/',  help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory name to save the fig')
    parser.add_argument('--save_dir', type=str, default='results/', help='Directory name to save the test result')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --log_dir
    check_folder(args.log_dir)
    # --save_dir
    check_folder(args.save_dir)
    
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def train(args, sess):
    print("Training...")
    input_ = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3])
    gt_img = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, args.out_dim])
    lr = tf.placeholder(tf.float32, name='learning_rate')

    model = VAE_UNet(args)
    z_mean, z_logvar, out_img = model.build(input_=input_, reuse=False)

    reg_loss, rec_loss, kl_loss, vae_loss = vae_criterion(out_img, gt_img, z_mean, z_logvar)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(vae_loss)

    var_list = tf.trainable_variables()

    saver = tf.train.Saver(var_list=var_list, max_to_keep=3)
    cur_lr = args.lr
    print("Learning rate:", cur_lr)

    src_data_path = args.data_path + args.mode.split('_')[0] + '/'
    dst_data_path = args.data_path + args.mode.split('_')[-1] + '/'
    src_list, gt_list = load_data_list(src_data_path, dst_data_path)

    data_len = len(src_list)
    iter = int(data_len / args.batch_size)
    if data_len % args.batch_size is not 0:
        iter += 1

    tf.global_variables_initializer().run()

    if args.continue_train:
        model_file = tf.train.latest_checkpoint(args.checkpoint_dir)
        saver.restore(sess, model_file)
        print("Successed load model paras " + model_file + '!')


    for i in range(0, args.max_iter):
        if i < iter:
            input_batch_list = src_list[i*args.batch_size: (i+1)*args.batch_size]
            gt_batch_list = gt_list[i*args.batch_size: (i+1)*args.batch_size]
        else:
            input_batch_list = src_list[i*args.batch_size:]
            gt_batch_list = gt_list[i*args.batch_size:]
            src_list, gt_list = random_list(src_list, gt_list)


        input_batch, gt_batch = load_batch(input_batch_list, gt_batch_list, args.img_size)

        _, l_reg, l_rec, l_kl, l_vae = sess.run([optim, reg_loss, rec_loss, kl_loss, vae_loss], feed_dict={input_:input_batch, gt_img:gt_batch, lr:cur_lr})

        print("Iter:%d, reg_loss:%f, rec_loss:%f, kl_loss:%f, vae_loss:%f" % (i, l_reg, l_rec, l_kl, l_vae))

        if i % 2000 == 0:
            saver.save(sess, args.checkpoint_dir + '/model_{}.ckpt'.format(str(i)))
            img_i, img_g, img_o = sess.run([input_, gt_img, out_img], feed_dict={input_:input_batch, gt_img:gt_batch})
            img_i = (img_i + 1) / 2
            fig = plot(img_i, args.img_size)
            plt.savefig('{}/src_{}.png'.format(args.log_dir, str(i).zfill(5)), bbox_inches='tight')
            img_g = (img_g + 1) / 2
            fig = plot(img_g, args.img_size)
            plt.savefig('{}/gt_{}.png'.format(args.log_dir, str(i).zfill(5)), bbox_inches='tight')
            img_o = (img_o + 1) / 2
            fig = plot(img_o, args.img_size)
            plt.savefig('{}/out_{}.png'.format(args.log_dir, str(i).zfill(5)), bbox_inches='tight')

    saver.save(sess, args.checkpoint_dir + '/model_{}.ckpt'.format(str(args.max_iter)))

def test(args, sess):
    print("Testing...")
    model = VAE_UNet(args)
    input_ = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3])
    _, _, out_img = model.build(input_=input_, reuse=False)

    var_list = tf.trainable_variables()

    saver = tf.train.Saver(var_list=var_list)
    test_datapath = args.test_data_path

    tf.global_variables_initializer().run()
    saver.restore(sess, args.pretrained_model_path)
    print("Successed load model paras " + args.pretrained_model_path + '!')

    for pic_name in os.listdir(test_datapath):
        print("pic_name:", pic_name)
        img_name = test_datapath+pic_name
        src_img = load_image(img_name, args.img_size)
        src_img = np.array(src_img).astype(np.float32)[None, :, :, :]
        img_o = sess.run(out_img, feed_dict={input_: src_img})
        img_o = (img_o[0] + 1) / 2
        save_path = args.save_dir + args.mode.split('_')[-1] + '/'
        check_folder(save_path)
        save_image(img_o, save_path+pic_name)

if __name__ == '__main__':
    args = parse_args()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if args.phase == 'train':
            train(args, sess=sess)
        elif args.phase == 'test':
            test(args, sess=sess)
