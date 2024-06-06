from pathlib import Path
import os
import tarfile
import csv
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


ALL_PARTICIPANT_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 37]
SLIPPAGE_FRAMES = 3


class EpicKitchensDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
            # Add other transformations if needed
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        frames = [self.transform(Image.open(frame_path)) for frame_path in sample['frames']]  # Apply transformation
        # for each frame print the shape
        for frame in frames:
            print(frame.shape)
        narration = sample['narration']
        return frames, narration
    

class EpicKitchensDataLoader:

    def __init__(self, output_directory, frames, participant_numbers = None, video_id = None, batch_size=64):
        
        if participant_numbers is None:
            participant_numbers = ALL_PARTICIPANT_NUMBERS
        
        self.participant_numbers = participant_numbers
        self.video_id = video_id
        self.frames = frames
        self.cwd = Path.cwd()
        self.output_directory = self.cwd / output_directory
        self.batch_size = batch_size
        self.dataset = []


    def check_data_exists_and_download_if_not(self):
        for participant in self.participant_numbers:
            # numher should be 2 digits so add a leading 0 if it is a single digit
            if participant < 10:
                participant_formatted = f"0{participant}"
            else:
                participant_formatted = participant
            participant_folder = self.output_directory/ "EPIC-KITCHENS" / f"P{participant_formatted}"

            if not participant_folder.exists():                # check if video_id is provided if so download only that video, else download all videos
                self.download_data(participant)

            # elif self.video_id:
            #     # check if video_id is provided if so download only that video, else download all videos
            #     # video id should be 2 digits so add a leading 0 if it is a single digit
            #     if self.video_id < 10:
            #         video_id_formatted = f"0{self.video_id}"
            #     else:
            #         video_id_formatted = self.video_id
            #     if not (participant_folder/ "rgb_frames" / f"P{participant}_{video_id_formatted}").exists():
            #         self.download_data(participant)

            print(f"Participant {participant} complete")
    

    def download_data(self,participant):

        ### have to run the following python command to download the data
        # python epic_downloader.py --rgb-frames --extension-only --participants {participant} --output_path {self.output_directory}
        if participant < 10:
            participant_formatted = f"0{participant}"
        else:
            participant_formatted = participant

        # if self.video_id:
        #     # construct the video id from the participant number and video id, noting for both single digit numbers we need to add a leading 0
        #     if self.video_id < 10:
        #         self.video_id = f"0{self.video_id}"
        #     video_id = f"P{participant_formatted}_{self.video_id}"
        #     command = f"python epic_downloader.py --rgb-frames --extension-only --participants P{participant} --specific-videos {video_id} --output_path {self.output_directory} --train"

        # else:
        command = f"python epic_downloader.py --rgb-frames --participants P{participant_formatted} --output_path {self.output_directory} --train"

        # RUN THE COMMAND
        print(f"Downloading data for participant {participant} {self.video_id if self.video_id else ''}")
        print(command)
        os.system(command)


    def untar_data(self,participants):

        if type(participants) != list:
            participants = [participants]
        
        for participant in participants:

            if participant < 10:
                participant_formatted = f"0{participant}"
            else:
                participant_formatted = participant
            participant_folder = self.output_directory/ "EPIC-KITCHENS" / f"P{participant_formatted}"/"rgb_frames"
            # check if any files are tar files and untar them
            try:
                # print directory name and contents
                print(participant_folder)
                print(list(participant_folder.iterdir()))

                
                for file in participant_folder.iterdir():
                    print(file)
                    if file.suffix == ".tar":
                        # output_folder file name without the .tar extension
                        with tarfile.open(file, "r:") as tar:
                            tar.extractall(path=participant_folder/file.stem)
                        
                        # delete the tar file
                        file.unlink()
            except:
                print(f"No tar files found for participant {participant}")


    def load_csv_data(self, csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                participant = int(row['participant_id'][1:])
                if participant in self.participant_numbers:
                    video_id = int(row['video_id'].split('_')[1])
                    if self.video_id is None or video_id == self.video_id:
                        start_frame = int(row['start_frame'])
                        stop_frame = int(row['stop_frame'])
                        if stop_frame - start_frame + 1 >= self.frames:
                            self.process_row(row)

    
    def process_row(self, row):
        participant_id = row['participant_id']
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        step = (stop_frame - start_frame) // (self.frames - 1)
        frames_to_check = [start_frame + i * step for i in range(self.frames)]
        adjusted_frames = self.check_frames_exist(participant_id, video_id, frames_to_check)
        if adjusted_frames:
            frame_paths = [self.get_frame_path(participant_id, video_id, frame) for frame in adjusted_frames]
            self.dataset.append({'frames': frame_paths, 'narration': row['narration']})


    def check_frames_exist(self, participant_id, video_id, frames):
        adjusted_frames = []
        for frame in frames:
            frame_path = self.get_frame_path(participant_id, video_id, frame)
            # print(frame)
            # print(frame_path)
            if not frame_path.exists():
                found = False
                for offset in range(1, SLIPPAGE_FRAMES + 1): # slippage of N frames
                    frame_path = self.get_frame_path(participant_id, video_id, frame - offset)
                    if frame_path.exists():
                        found = True
                        adjusted_frames.append(frame - offset)
                        break
                if not found:
                    return False
            else:
                adjusted_frames.append(frame)
        return adjusted_frames
    

    def get_frame_path(self, participant_id, video_id, frame):
        frame_str = f"frame_{frame:010d}.jpg"
        return self.output_directory / "EPIC-KITCHENS" / participant_id / "rgb_frames" / video_id / frame_str

    def get_dataloader(self):
        return DataLoader(EpicKitchensDataset(self.dataset), batch_size=self.batch_size, shuffle=True)

if __name__ == "__main__":
    # parse arguments
    # User can pass participant number (if not, default to 1)
    # User can pass video id (optional, this will just be a number)
    # User can pass output directory (this should be relative to where the script is run)
    # User can pass batch size (default to 64)

    # epicKitchenDataLoader = EpicKichensDataLoader(output_directory="data",participant_numbers=[1])
    # epicKitchenDataLoader = EpicKichensDataLoader(output_directory="data",participant_numbers=[1],video_id=3)
    # epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",participant_numbers=[4], video_id=1)
    
    #### FOR KOBI STEP BY STEP

    """
    1) Pick a participant number from the list of participant numbers 
     - i.e. 2 ,
    """
    
    ### Utility function to help pick a participant number
    def count_participant_ids(csv_path):
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            participant_ids = {}
            for row in reader:
                participant_id = row['participant_id']
                participant_ids[participant_id] = participant_ids.get(participant_id, 0) + 1

        # print participant_ids sorted by count in ascending order
        for key, value in sorted(participant_ids.items(), key=lambda item: item[1]):
            print(f"{key}: {value}")

    count_participant_ids("EPIC_100_train.csv")
    
    
    # epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",frames=20,participant_numbers=[2], video_id=101)
    # epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",frames=20,participant_numbers=[16,17,14,19,21,15,13,9,31,5,37])
    # epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",frames=20,participant_numbers=[2])
    pariticipants = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 37]
    epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",frames=20,participant_numbers=pariticipants,batch_size=1)
    epicKitchenDataLoader.check_data_exists_and_download_if_not()
    epicKitchenDataLoader.untar_data(pariticipants)
    # epicKitchenDataLoader.load_csv_data("EPIC_100_train.csv")
    # dataloader = epicKitchenDataLoader.get_dataloader()

    # for batch in dataloader:
    #     frames, narrations = batch
    #     # print(frames)
    #     print(narrations)



