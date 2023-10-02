#minimialist pytorch dataset for operating with the 50Salads data.
#loosely based on TSN implementation. https://github.com/yjxiong/tsn-pytorch/blob/master/dataset.py
import torch.utils.data as data
from PIL import Image
import os
import numpy as np


class VideoRecord(object):
	def __init__(self, row):
		self._data = row

	@property
	def name(self): 
		return self._data[0] #return the video 

	@property
	def start_frame(self):
		return int(self._data[1]) #return the start frame of the segment

	@property
	def end_frame(self):
		return int(self._data[2]) #return the end frame of the segment

	@property
	def class_label(self):
		return int(self._data[4]) #return the label of the segment.


class SaladsDataSet(data.Dataset):

	def __init__(self,rootpath,label_path,split,t_stride=8,num_frames=8,sampling_type='dense',mode='train',transforms=None):
		self.rootpath = rootpath
		self.label_path = label_path

		self.mode = mode
		assert self.mode in ['train','test'], "dataset mode must be either train or test"

		self.t_stride = t_stride
		self.num_frames = num_frames
		self.image_tmpl='{:06d}.jpg'

		self.sampling_type = sampling_type
		assert self.sampling_type in ['dense','surround'], "sample type must be either dense or surround"

		self.transforms = transforms #spatial transforms we wish to perform to the images

		#construct the video list for our given split
		self._parse_segment_labels(split)


	def _parse_segment_labels(self,split):

		split_path = 'Splits/Split{}/{}.txt'.format(split,self.mode) #get the folders for a given split.

		#open split path and get names of all the segments...
		with open(split_path, 'r') as f:
			vids_in_split = f.read().split('\n')
		
		all_labels = []
		for vid_id in vids_in_split:
			vid_name = '{}rgb-{}.txt'.format(self.label_path,vid_id)
			tmp = [x.strip().split(',') for x in open(vid_name)]
			for label in tmp:
				label.insert(0,str(vid_id))
				all_labels.append(label)

		self.video_list = [VideoRecord(item) for item in all_labels]

	
	def _get_max_idx(self,record):
		"""
		return the maximum idx of the frame for a given video
		"""
		return len(os.listdir('{}rgb-{}/'.format(self.rootpath,record.name)))

	def _get_train_indices(self,record):
		"""
		get the sample train indices for a given [X,y] pair using either surround or dense sampling
		"""
		clip_length = self.t_stride*self.num_frames

		if self.sampling_type == 'surround':
			sample_pos = np.random.randint(int(record.start_frame - (clip_length/2)),int(record.end_frame - (clip_length/2))) #surround sampling as implemented in https://arxiv.org/abs/2211.13694
			indices = np.asarray([(idx * self.t_stride + sample_pos) for idx in range(self.num_frames)])

			#crop indicies which are beyond the number of frames in the full video - occurs rarely. 
			max_idx = self._get_max_idx(record)
			indices[indices<1] = 1
			indices[indices>max_idx] = max_idx

		elif self.sampling_type == 'dense': #dense sampling as implemented in most works.
			clip_length = self.t_stride*self.num_frames #the desired clip length we want to extract.
			seg_length = record.end_frame - record.start_frame

			if seg_length-clip_length > self.t_stride:
				sample_pos = np.random.randint(record.start_frame,int(record.end_frame-clip_length))
			else: #using surround sampling, to select initial index, but will clip the indicies...
				sample_pos = np.random.randint(int(record.start_frame - (clip_length/2)),int(record.end_frame - (clip_length/2))) 

			indices = np.asarray([(idx * self.t_stride + sample_pos) for idx in range(self.num_frames)])
			indices[indices<record.start_frame] = record.start_frame
			indices[indices>record.end_frame] = record.end_frame 

		return indices


	def _get_test_indices(self,record):
		"""
		get the sample indices for a given [X,y] pair - we only implement a single central crop, other works may utilise multiple temporal and spatial crops.
		"""
		clip_length = self.t_stride*self.num_frames
		sample_pos = (record.end_frame - (record.end_frame - record.start_frame)/2 - (clip_length/2))

		indices = np.asarray([(idx * self.t_stride + sample_pos) for idx in range(self.num_frames)])

		if self.sampling_type == 'dense':
        #clip the indicies if dense sampling - otherwise leave them as we have default to suroround sampling.
			indices[indices<record.start_frame] = record.start_frame
			indices[indices>record.end_frame] = record.end_frame 

		return indices


	def __getitem__(self,index):
		record = self.video_list[index] #get the data for a given example in our list.

		if self.mode == 'train':
			segment_indices = self._get_train_indices(record)
		elif self.mode == 'test':
			segment_indices = self._get_test_indices(record)

		return self._get(record,segment_indices)

	def _get(self,record,indices):
		#main function to retrieve images and apply transforms.

		images = list()

		for i, seg_ind in enumerate(indices):
			img_name = 'rgb-{}'.format(record.name)
			img_path = os.path.join(self.rootpath,img_name,self.image_tmpl.format(int(seg_ind)))

			try:
				seg_imgs = [Image.open(img_path).convert('RGB')]
				images.extend(seg_imgs)
			except:
				img_path = os.path.join(self.rootpath,img_name,self.image_tmpl.format(int(seg_ind-1))) #try previous image.
				seg_imgs = [Image.open(img_path).convert('RGB')]
				images.extend(seg_imgs)

		if self.transforms is not None:
			processed_data = self.transforms(images)
		else:
			processed_data = images

		return processed_data,record.class_label

		#path needs to be updated from the name we have.

	def __len__(self):
		return len(self.video_list)



if __name__ == '__main__':

	#code to test the loader with some simple transformations 
	import torchvision
	from transforms import *

	train_trans = torchvision.transforms.Compose([
			GroupScale(256),
			GroupRandomCrop(224),
			Stack(),
			ToTorchFormatTensor(),
			GroupNormalize(
				mean=[.485, .456, .406],
				std=[.229, .224, .225]
			)]
		)

	test_trans = torchvision.transforms.Compose([
		GroupScale(256),
		GroupCenterCrop(224),
		Stack(),
		ToTorchFormatTensor(),
		GroupNormalize(
			mean=[.485, .456, .406],
			std=[.229, .224, .225]
		)]
	)


	train_data = SaladsDataSet(rootpath='rgb_images/',label_path='Frame_labels/',split=1,t_stride=8,num_frames=8,sampling_type='surround',mode='train',transforms=train_trans)
	test_data = SaladsDataSet(rootpath='rgb_images/',label_path='Frame_labels/',split=1,t_stride=8,num_frames=8,sampling_type='surround',mode='test',transforms=test_trans)

	for i,example in enumerate(test_data):
		data,label = example
		print(data,label)
