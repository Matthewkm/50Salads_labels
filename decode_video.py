import os
import argparse

def arg_parse():
	parser = argparse.ArgumentParser(description='Arguments for decoding 50Salads video data')
	parser.add_argument("--vidDir",default='rgb')
	parser.add_argument("--outDir",default='rgb_images')
	return parser.parse_args()

args = arg_parse()

#go through the videos in rgb:
for vid_file in os.listdir(args.vidDir):
	
	#get save name of decoded video
	video_path = '{}/{}'.format(args.vidDir,vid_file)
	video_name = os.path.basename(video_path)[:-4] 
	new_folder = os.path.join(args.outDir,video_name)

	if os.path.isdir(new_folder): #if folder already exists
		if os.listdir(new_folder) == 0: #if folder is empty, decode the video into it.
			print('decoding the video: {}'.format(vid_file))
			bashCommand = 'ffmpeg -i "{}" -vf scale=-1:480 -q:v 1 "{}/%06d.jpg'.format(video_path,new_folder,video_name) #create clips.
			os.system(bashCommand)

		else:
			print('A folder named {} already exists, ignoring this video.'.format(vid_file))

	else: #need to decode the video into a folder of images - makes life easier...
		print('decoding the video: {}'.format(vid_file))
		os.mkdir(new_folder)
		bashCommand = 'ffmpeg -i "{}" -vf scale=-1:480 -q:v 1 "{}/%06d.jpg'.format(video_path,new_folder,video_name) #create clips.
		os.system(bashCommand)
