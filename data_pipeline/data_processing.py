from pathlib import Path
import os
import tarfile

ALL_PARTICIPANT_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35, 37]

class EpicKitchensDataLoader:

    def __init__(self, output_directory, participant_numbers = None, video_id = None , batch_size=64):
        
        if participant_numbers is None:
            participant_numbers = ALL_PARTICIPANT_NUMBERS
        
        self.participant_numbers = participant_numbers
        self.video_id = video_id

        self.cwd = Path.cwd()
        self.output_directory = self.cwd / output_directory
        self.batch_size = batch_size


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

            elif self.video_id:
                # check if video_id is provided if so download only that video, else download all videos
                # video id should be 2 digits so add a leading 0 if it is a single digit
                if self.video_id < 10:
                    video_id_formatted = f"0{self.video_id}"
                else:
                    video_id_formatted = self.video_id
                if not (participant_folder/ "rgb_frames" / f"P{participant}_{video_id_formatted}").exists():
                    self.download_data(participant)

            self.untar_data(participant)

            print(f"Participant {participant} complete")
    
    def download_data(self,participant):

        ### have to run the following python command to download the data
        # python epic_downloader.py --rgb-frames --extension-only --participants {participant} --output_path {self.output_directory}
        if self.video_id:
            # construct the video id from the participant number and video id, noting for both single digit numbers we need to add a leading 0
            if participant < 10:
                participant_formatted = f"0{participant}"
            if self.video_id < 10:
                self.video_id = f"0{self.video_id}"
            video_id = f"P{participant_formatted}_{self.video_id}"
            command = f"python epic_downloader.py --rgb-frames --extension-only --participants {participant} --specific-videos {video_id} --output_path {self.output_directory} --train"
        else:
            command = f"python epic_downloader.py --rgb-frames --extension-only --participants {participant} --output_path {self.output_directory} --train"

        # RUN THE COMMAND
        print(f"Downloading data for participant {participant} {self.video_id if self.video_id else ''}")
        print(command)
        os.system(command)


    def untar_data(self,participant):
        if participant < 10:
            participant_formatted = f"0{participant}"
        else:
            participant_formatted = participant
        participant_folder = self.output_directory/ "EPIC-KITCHENS" / f"P{participant_formatted}"/"rgb_frames"
        # check if any files are tar files and untar them
        try:
            for file in participant_folder.iterdir():
                if file.suffix == ".tar":
                    # output_folder file name without the .tar extension
                    with tarfile.open(file, "r:") as tar:
                        tar.extractall(path=participant_folder/file.stem)
                    
                    # delete the tar file
                    file.unlink()
        except:
            print(f"No tar files found for participant {participant}")



if __name__ == "__main__":
    # parse arguments
    # User can pass participant number (if not, default to 1)
    # User can pass video id (optional, this will just be a number)
    # User can pass output directory (this should be relative to where the script is run)
    # User can pass batch size (default to 64)

    # epicKitchenDataLoader = EpicKichensDataLoader(output_directory="data",participant_numbers=[1])
    # epicKitchenDataLoader = EpicKichensDataLoader(output_directory="data",participant_numbers=[1],video_id=3)
    # epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",participant_numbers=[4], video_id=1)
    epicKitchenDataLoader = EpicKitchensDataLoader(output_directory="data",participant_numbers=[6], video_id=1)
    epicKitchenDataLoader.check_data_exists_and_download_if_not()




##### IGNORE BELOW
#    P01_01 (note the participant number may omit the leading 0, if number is a single digit add it back))    
