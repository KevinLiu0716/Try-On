# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13i68X4oozjl3KjeY77HZIj7HHm_KQ9M_
"""

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init

# INIT_TYPE = "kaiming"

# initialize network weight
def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
	# print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def featureL2Norm(feature):
	epsilon = 1e-6
	norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
	return torch.div(feature,norm)

# loading the resnet-18 model and removing the fully connected layers to preserve the spatial dimension details
class FeatureExtraction(nn.Module):
	def __init__(self, normalization=True, use_cuda=True):
		super(FeatureExtraction, self).__init__()
		self.normalization = normalization
		self.use_cuda = use_cuda
		self.model = models.resnet18(pretrained=False)
		self.model = nn.Sequential(*list(self.model.children())[:-2])
		init_weights(self.model, init_type="normal")
		
	def forward(self, image_batch):
		features = self.model(image_batch)
		if self.normalization:
			features = featureL2Norm(features)
		return features

	def freeze_layers(self):
		for param in self.model.parameters():
			param.requires_grad = False

	def unfreeze_layers(self):
		for param in self.model.parameters():
			param.requires_grad = True
		ct = 0
		for name, child in self.model.named_children():
			ct += 1
			if ct < 6:
				for name2, parameters in child.named_parameters():
					parameters.requires_grad = False

# model to lower the number of channels to 3 for feeding the agnostic model with key points to the resnet-18 model
class ReduceNumberOfChannels(nn.Module):
	def __init__(self, use_cuda=True):
		super(ReduceNumberOfChannels, self).__init__()
		self.use_cuda = use_cuda
		self.model = nn.Sequential(
            # input is 24 x 224 x 224
            nn.Conv2d(6, 3, 1, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(3))

	def forward(self, image_batch):
		return self.model(image_batch)


# the correlation module 
class FeatureCorrelation(nn.Module):
	def __init__(self,normalization=True,matching_type='correlation'):
		super(FeatureCorrelation, self).__init__()
		self.normalization = normalization
		self.matching_type=matching_type
		self.ReLU = nn.ReLU()
	
	def forward(self, feature_A, feature_B):
		b,c,h,w = feature_A.size()
		if self.matching_type=='correlation':
			# reshape features for matrix multiplication
			feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
			feature_B = feature_B.view(b,c,h*w).transpose(1,2)
			# perform matrix mult.
			feature_mul = torch.bmm(feature_B,feature_A)
			# indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
			correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
			if self.normalization:
				correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
			return correlation_tensor

		if self.matching_type=='subtraction':
			return feature_A.sub(feature_B)
		
		if self.matching_type=='concatenation':
			return torch.cat((feature_A,feature_B),1)

# the regression module to get the theta outputs for transformation
class FeatureRegression(nn.Module):
	def __init__(self,output_dim=6, use_cuda=True):
		super(FeatureRegression, self).__init__()
		self.output_dim = output_dim

		# self.conv1 = nn.Conv2d(196, 64, kernel_size=3)
		# self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(49, 32, kernel_size=3)
		self.bn2 = nn.BatchNorm2d(32)
		# self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=0)
		# self.bn3 = nn.BatchNorm2d(32)
		# self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=0)
		# self.bn4 = nn.BatchNorm2d(16)
		# self.linear1 = nn.Linear(3200, 512)
		# self.linear2 = nn.Linear(512, 6)

		# Regressor for the 3 * 2 affine matrix
		self.fc_loc = nn.Sequential(
			nn.Linear(800, 256),
			nn.ReLU(True),
			nn.Linear(256, 128),
			nn.ReLU(True),
			nn.Linear(128, self.output_dim)
		)

		# Initialize the weights/bias with identity transformation
		if(self.output_dim == 6):
			self.fc_loc[4].weight.data.zero_()
			self.fc_loc[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
		# elif(self.output_dim == 18):
		# 	self.fc_loc[4].weight.data.zero_()
		# 	self.fc_loc[4].bias.data.copy_(torch.tensor([0]*18, dtype=torch.float))



	def forward(self, x):
		# x = self.conv1(x)
		# x = self.bn1(x)
		# x = nn.ReLU()(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = nn.ReLU()(x)
		# x = self.conv3(x)
		# x = self.bn3(x)
		# x = nn.ReLU()(x)
		# x = self.conv4(x)
		# x = self.bn4(x)
		# x = nn.ReLU()(x)
		x = x.reshape(x.size(0), -1)
		# x = self.linear1(x)
		# x = nn.ReLU()(x)
		# x = self.linear2(x)
		x = self.fc_loc(x)
		x = F.relu(x)

		return x


class GenerateTheta(nn.Module):
	def __init__(self,geometric_model="affine", tps_grid_size=3, use_cuda=True):
		super(GenerateTheta, self).__init__()
		self.geometric_model = geometric_model
		self.tps_grid_size = tps_grid_size
		self.use_cuda = use_cuda
		self.FeatureExtraction = FeatureExtraction()
		self.ReduceNumberOfChannels = ReduceNumberOfChannels()
		self.FeatureCorrelation = FeatureCorrelation()
		if(self.geometric_model == "affine"):
			self.FeatureRegression = FeatureRegression(output_dim=6)
		elif(self.geometric_model == "tps"):
			self.FeatureRegression = FeatureRegression(output_dim=2*(self.tps_grid_size ** 2))
		if use_cuda:
			self.ReduceNumberOfChannels = self.ReduceNumberOfChannels
			self.FeatureExtraction = self.FeatureExtraction
			self.FeatureCorrelation = self.FeatureCorrelation
			self.FeatureRegression = self.FeatureRegression
		self.ReLU = nn.ReLU(inplace=True)

	def freeze_layers(self):
		self.FeatureExtraction.freeze_layers()

	def unfreeze_layers(self):
		self.FeatureExtraction.unfreeze_layers()

	def forward(self, product_image_batch, model_image_batch):
		# feature extraction
		feature_A = self.FeatureExtraction(product_image_batch)
		model_image_batch = self.ReduceNumberOfChannels(model_image_batch)
		feature_B = self.FeatureExtraction(model_image_batch)
		feature_A = featureL2Norm(feature_A)
		feature_B = featureL2Norm(feature_B)

		# feature correlation
		correlation = self.FeatureCorrelation(feature_A, feature_B)
		# regression for transformation parameters
		theta = self.FeatureRegression(correlation)

		return theta

# standard import

# third party imports
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# local imports

class GeometricTransformer(nn.Module):
	def __init__(self, geometric_model="affine", tps_grid_size=3):
		super(GeometricTransformer, self).__init__()
		self.geometric_model = geometric_model
		self.tps_grid_size = tps_grid_size
		if(self.geometric_model == "affine"):
			self.grid_gen = AffineGridGen()
		elif(self.geometric_model == "tps"):
			self.grid_gen = TpsGridGen(grid_size = self.tps_grid_size)

	def forward(self, image_batch, theta):
		grid = self.grid_gen(theta)
		warped_image_batch = F.grid_sample(image_batch, grid)
		return warped_image_batch

class AffineGridGen(nn.Module):
	def __init__(self, out_h=224, out_w=224, out_ch = 3, use_cuda=True):
		super(AffineGridGen, self).__init__()        
		self.out_h = out_h
		self.out_w = out_w
		self.out_ch = out_ch
		
	def forward(self, theta):
		b=theta.size()[0]
		if not theta.size()==(b,2,3):
			theta = theta.view(-1,2,3)
		theta = theta.contiguous()
		batch_size = theta.size()[0]
		out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
		return F.affine_grid(theta, out_size)

class TpsGridGen(nn.Module):
	def __init__(self, out_h=224, out_w=224, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
		super(TpsGridGen, self).__init__()
		self.out_h, self.out_w = out_h, out_w
		self.reg_factor = reg_factor
		self.use_cuda = use_cuda

		# create grid in numpy
		self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
		# sampling grid with dim-0 coords (Y)
		self.grid_X,self.grid_Y = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
		# grid_X,grid_Y: size [1,H,W,1,1]
		self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
		self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
		if use_cuda:
			self.grid_X = self.grid_X
			self.grid_Y = self.grid_Y

		# initialize regular grid for control points P_i
		if use_regular_grid:
			axis_coords = np.linspace(-1,1,grid_size)
			self.N = grid_size*grid_size
			P_Y,P_X = np.meshgrid(axis_coords,axis_coords)
			P_X = np.reshape(P_X,(-1,1)) # size (N,1)
			P_Y = np.reshape(P_Y,(-1,1)) # size (N,1)
			P_X = torch.FloatTensor(P_X)
			P_Y = torch.FloatTensor(P_Y)
			self.P_X_base = P_X.clone()
			self.P_Y_base = P_Y.clone()
			self.Li = self.compute_L_inverse(P_X,P_Y).unsqueeze(0)
			self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
			self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4)
			if use_cuda:
				self.P_X = self.P_X
				self.P_Y = self.P_Y
				self.P_X_base = self.P_X_base
				self.P_Y_base = self.P_Y_base

	def forward(self, theta):
		gpu_id = theta.get_device()
		self.grid_X = self.grid_X.to(gpu_id)
		self.grid_Y = self.grid_Y.to(gpu_id)
		self.P_X = self.P_X.to(gpu_id)
		self.P_Y = self.P_Y.to(gpu_id)
		self.P_X_base = self.P_X_base.to(gpu_id)
		self.P_Y_base = self.P_Y_base.to(gpu_id)
		self.Li = self.Li.to(gpu_id) 
		warped_grid = self.apply_transformation(theta,torch.cat((self.grid_X,self.grid_Y),3))
		return warped_grid
		
	def compute_L_inverse(self,X,Y):
		N = X.size()[0] # num of points (along dim 0)
		# construct matrix K
		Xmat = X.expand(N,N)
		Ymat = Y.expand(N,N)
		P_dist_squared = torch.pow(Xmat-Xmat.transpose(0,1),2)+torch.pow(Ymat-Ymat.transpose(0,1),2)
		P_dist_squared[P_dist_squared==0]=1 # make diagonal 1 to avoid NaN in log computation
		K = torch.mul(P_dist_squared,torch.log(P_dist_squared))
		# construct matrix L
		O = torch.FloatTensor(N,1).fill_(1)
		Z = torch.FloatTensor(3,3).fill_(0)       
		P = torch.cat((O,X,Y),1)
		L = torch.cat((torch.cat((K,P),1),torch.cat((P.transpose(0,1),Z),1)),0)
		self.Li = torch.inverse(L)
		if self.use_cuda:
			self.Li = self.Li
		return self.Li
		
	def apply_transformation(self,theta,points):
		if theta.dim()==2:
			theta = theta.unsqueeze(2).unsqueeze(3)
		# points should be in the [B,H,W,2] format,
		# where points[:,:,:,0] are the X coords  
		# and points[:,:,:,1] are the Y coords  
		
		# input are the corresponding control points P_i
		batch_size = theta.size()[0]
		# split theta into point coordinates
		Q_X=theta[:,:self.N,:,:].squeeze(3)
		Q_Y=theta[:,self.N:,:,:].squeeze(3)

		Q_X = Q_X + self.P_X_base.expand_as(Q_X)
		Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
		
		# get spatial dimensions of points
		points_b = points.size()[0]
		points_h = points.size()[1]
		points_w = points.size()[2]
		
		# repeat pre-defined control points along spatial dimensions of points to be transformed
		P_X = self.P_X.expand((1,points_h,points_w,1,self.N))
		P_Y = self.P_Y.expand((1,points_h,points_w,1,self.N))
		
		# compute weigths for non-linear part
		W_X = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_X)
		W_Y = torch.bmm(self.Li[:,:self.N,:self.N].expand((batch_size,self.N,self.N)),Q_Y)
		# reshape
		# W_X,W,Y: size [B,H,W,1,N]
		W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
		W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
		# compute weights for affine part
		A_X = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_X)
		A_Y = torch.bmm(self.Li[:,self.N:,:self.N].expand((batch_size,3,self.N)),Q_Y)
		# reshape
		# A_X,A,Y: size [B,H,W,1,3]
		A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
		A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,points_h,points_w,1,1)
		
		# compute distance P_i - (grid_X,grid_Y)
		# grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
		points_X_for_summation = points[:,:,:,0].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,0].size()+(1,self.N))
		points_Y_for_summation = points[:,:,:,1].unsqueeze(3).unsqueeze(4).expand(points[:,:,:,1].size()+(1,self.N))
		
		if points_b==1:
			delta_X = points_X_for_summation-P_X
			delta_Y = points_Y_for_summation-P_Y
		else:
			# use expanded P_X,P_Y in batch dimension
			delta_X = points_X_for_summation-P_X.expand_as(points_X_for_summation)
			delta_Y = points_Y_for_summation-P_Y.expand_as(points_Y_for_summation)
			
		dist_squared = torch.pow(delta_X,2)+torch.pow(delta_Y,2)
		# U: size [1,H,W,1,N]
		dist_squared[dist_squared==0]=1 # avoid NaN in log computation
		U = torch.mul(dist_squared,torch.log(dist_squared)) 
		
		# expand grid in batch dimension if necessary
		points_X_batch = points[:,:,:,0].unsqueeze(3)
		points_Y_batch = points[:,:,:,1].unsqueeze(3)
		if points_b==1:
			points_X_batch = points_X_batch.expand((batch_size,)+points_X_batch.size()[1:])
			points_Y_batch = points_Y_batch.expand((batch_size,)+points_Y_batch.size()[1:])
		
		points_X_prime = A_X[:,:,:,:,0]+ \
					   torch.mul(A_X[:,:,:,:,1],points_X_batch) + \
					   torch.mul(A_X[:,:,:,:,2],points_Y_batch) + \
					   torch.sum(torch.mul(W_X,U.expand_as(W_X)),4)
					
		points_Y_prime = A_Y[:,:,:,:,0]+ \
					   torch.mul(A_Y[:,:,:,:,1],points_X_batch) + \
					   torch.mul(A_Y[:,:,:,:,2],points_Y_batch) + \
					   torch.sum(torch.mul(W_Y,U.expand_as(W_Y)),4)
		
		return torch.cat((points_X_prime,points_Y_prime),3)

# standard library imports

# third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


#---------------------- basic convolution blocks ------------------------#
# 3x3 convolution with given input and output info
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
	return nn.Conv2d(
		in_channels,
		out_channels,
		kernel_size=3,
		stride=stride,
		padding=padding,
		bias=bias,
		groups=groups)

# upsampling with other deconvolution or bilinear upsampling
def upconv2x2(in_channels, out_channels, mode='transpose'):
	if mode == 'transpose':
		return nn.ConvTranspose2d(
			in_channels,
			out_channels,
			kernel_size=2,
			stride=2)
	else:
		# out_channels is always going to be the same
		# as in_channels
		return nn.Sequential(
			nn.Upsample(mode='bilinear', scale_factor=2),
			conv1x1(in_channels, out_channels))

# 1x1 convolution to change the number of channels
def conv1x1(in_channels, out_channels, groups=1):
	return nn.Conv2d(
		in_channels,
		out_channels,
		kernel_size=1,
		groups=groups,
		stride=1)


# basic buliding block of the decoder part
class DownConv(nn.Module):
	"""
	A helper Module that performs 2 convolutions and 1 MaxPool.
	A ReLU activation follows each convolution.
	"""
	def __init__(self, in_channels, out_channels, pooling=True):
		super(DownConv, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.pooling = pooling

		self.conv1 = conv3x3(self.in_channels, self.out_channels)
		self.conv2 = conv3x3(self.out_channels, self.out_channels)

		if self.pooling:
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		before_pool = x
		if self.pooling:
			x = self.pool(x)
		return x, before_pool

# basic building block of the encoder part
class UpConv(nn.Module):
	"""
	A helper Module that performs 2 convolutions and 1 UpConvolution.
	A ReLU activation follows each convolution.
	"""
	def __init__(self, in_channels, out_channels, 
				 merge_mode='concat', up_mode='transpose'):
		super(UpConv, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.merge_mode = merge_mode
		self.up_mode = up_mode

		self.upconv = upconv2x2(self.in_channels, self.out_channels, 
			mode=self.up_mode)

		if self.merge_mode == 'concat':
			self.conv1 = conv3x3(
				2*self.out_channels, self.out_channels)
		else:
			# num of input channels to conv2 is same
			self.conv1 = conv3x3(self.out_channels, self.out_channels)
		self.conv2 = conv3x3(self.out_channels, self.out_channels)


	def forward(self, from_down, from_up):
		""" Forward pass
		Arguments:
			from_down: tensor from the encoder pathway
			from_up: upconv'd tensor from the decoder pathway
		"""
		from_up = self.upconv(from_up)
		if self.merge_mode == 'concat':
			x = torch.cat((from_up, from_down), 1)
		else:
			x = from_up + from_down
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return x


class UNet(nn.Module):
	""" `UNet` class is based on https://arxiv.org/abs/1505.04597
	The U-Net is a convolutional encoder-decoder neural network.
	Contextual spatial information (from the decoding,
	expansive pathway) about an input tensor is merged with
	information representing the localization of details
	(from the encoding, compressive pathway).
	Modifications to the original paper:
	(1) padding is used in 3x3 convolutions to prevent loss
		of border pixels
	(2) merging outputs does not require cropping due to (1)
	(3) residual connections can be used by specifying
		UNet(merge_mode='add')
	(4) if non-parametric upsampling is used in the decoder
		pathway (specified by upmode='upsample'), then an
		additional 1x1 2d convolution occurs after upsampling
		to reduce channel dimensionality by a factor of 2.
		This channel halving happens with the convolution in
		the tranpose convolution (specified by upmode='transpose')
	"""

	def __init__(self, num_classes, in_channels_1=3, in_channels_2=6, depth=5, 
				 start_filts=64, up_mode='transpose', 
				 merge_mode='concat', geometric_model="tps"):
		"""
		Arguments:
			in_channels: int, number of channels in the input tensor.
				Default is 3 for RGB images.
			depth: int, number of MaxPools in the U-Net.
			start_filts: int, number of convolutional filters for the 
				first conv.
			up_mode: string, type of upconvolution. Choices: 'transpose'
				for transpose convolution or 'upsample' for nearest neighbour
				upsampling.
		"""
		super(UNet, self).__init__()

		if up_mode in ('transpose', 'upsample'):
			self.up_mode = up_mode
		else:
			raise ValueError("\"{}\" is not a valid mode for "
							 "upsampling. Only \"transpose\" and "
							 "\"upsample\" are allowed.".format(up_mode))
	
		if merge_mode in ('concat', 'add'):
			self.merge_mode = merge_mode
		else:
			raise ValueError("\"{}\" is not a valid mode for"
							 "merging up and down paths. "
							 "Only \"concat\" and "
							 "\"add\" are allowed.".format(up_mode))

		# NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
		if self.up_mode == 'upsample' and self.merge_mode == 'add':
			raise ValueError("up_mode \"upsample\" is incompatible "
							 "with merge_mode \"add\" at the moment "
							 "because it doesn't make sense to use "
							 "nearest neighbour to reduce "
							 "depth channels (by half).")

		self.geometric_model = geometric_model

		self.num_classes = num_classes
		self.in_channels_1 = in_channels_1
		self.in_channels_2 = in_channels_2
		self.start_filts = start_filts
		self.depth = depth

		self.down_convs_1 = []
		self.down_convs_2 = []
		self.up_convs = []

		# create the encoder pathway and add to a list
		for i in range(depth):
			ins = self.in_channels_1 if i == 0 else outs
			outs = self.start_filts*(2**i)
			pooling = True if i < depth-1 else False

			down_conv = DownConv(ins, outs, pooling=pooling)
			self.down_convs_1.append(down_conv)

		for i in range(depth):
			ins = self.in_channels_2 if i == 0 else outs
			outs = self.start_filts*(2**i)
			pooling = True if i < depth-1 else False

			down_conv = DownConv(ins, outs, pooling=pooling)
			self.down_convs_2.append(down_conv)

		# create the decoder pathway and add to a list
		# - careful! decoding only requires depth-1 blocks

		outs = 2 * outs
		for i in range(depth-1):
			ins = outs
			outs = ins // 2
			up_conv = UpConv(ins, outs, up_mode=up_mode,
				merge_mode=merge_mode)
			self.up_convs.append(up_conv)

		self.conv_final = conv1x1(outs, self.num_classes)

		# add the list of modules to current module
		self.down_convs_1 = nn.ModuleList(self.down_convs_1)
		self.down_convs_2 = nn.ModuleList(self.down_convs_2)
		self.up_convs = nn.ModuleList(self.up_convs)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			init.xavier_normal(m.weight)
			init.constant(m.bias, 0)


	def reset_params(self):
		for i, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x1, x2, theta):
		encoder_outs_1 = []
		encoder_outs_2 = []
		encoder_outs = []

		for i, module in enumerate(self.down_convs_1):
			x1, before_pool = module(x1)
			encoder_outs_1.append(before_pool)
		
		for i, module in enumerate(self.down_convs_2):
			x2, before_pool = module(x2)
			encoder_outs_2.append(before_pool)
		
		for i, output in enumerate(encoder_outs_1):
			size = output.size()
			# object initialized for geometric transformation
			if(self.geometric_model == "affine"):
				grid_gen = AffineGridGen(out_h=size[3], out_w=size[3])
			else:
				grid_gen = TpsGridGen(out_h=size[3], out_w=size[3], grid_size=3)
			warped_grid = grid_gen(theta)
			encoder_outs_1[i] = F.grid_sample(encoder_outs_1[i], warped_grid)

		for i, output in enumerate(encoder_outs_1):
			encoder_outs.append(torch.cat((encoder_outs_1[i], encoder_outs_2[i]), 1))

		x = torch.cat((x1, x2), 1)

		for i, module in enumerate(self.up_convs):
			before_pool = encoder_outs[-(i+2)]
			x = module(before_pool, x)

		# No softmax is used. This means you need to use
		# nn.CrossEntropyLoss is your training script,
		# as this module includes a softmax already.
		x = self.conv_final(x)
		return x


class WUTON(nn.Module):
	def __init__(self):
		super().__init__()
		self.thetaGeneratorAffine = GenerateTheta(geometric_model="affine")
		self.thetaGeneratorTPS = GenerateTheta(geometric_model="tps", tps_grid_size=3)

		self.geo_tnf_affine = GeometricTransformer(geometric_model="affine")
		self.geo_tnf_tps = GeometricTransformer(geometric_model="tps", tps_grid_size=3)

		self.unet = UNet(3)

	def forward(self, gan_product_image_batch, model_agnostic_image_batch):

		theta_affine = self.thetaGeneratorAffine(gan_product_image_batch, model_agnostic_image_batch)
		affine_output = self.geo_tnf_affine(gan_product_image_batch, theta_affine)
		
		# tps transformation step
		theta_tps = self.thetaGeneratorTPS(affine_output, model_agnostic_image_batch)
		tps_output = self.geo_tnf_tps(affine_output, theta_tps)
		
		# unet transformation step
		unet_output = self.unet(gan_product_image_batch, model_agnostic_image_batch, theta_tps)
		return unet_output
