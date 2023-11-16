#!/bin/bash

#The following code was based on a script generated using langauge model GPT3.5 developed by OpenAI.
#The base code was generated using GPT-3.5 on October 30, 2023.



#Define the command you want to run
command_to_run="sbatch"

#Loop through all the .sh files in the current directory
for file in *.sh; do
	#Check if the file exists and is a regular file
	if [ -f "$file" ]; then
		#Run the command with the file as an argument
		$command_to_run "$file"
	fi
done

