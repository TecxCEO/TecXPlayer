#!/bin/bash

# Define the visual prompt for the menu
PS3="Please enter your choice (1-7): "

# List the menu options
options=("Run TecXPlayer" "Run" "Show System Info" "Update Packages" "Print Hello World" "Exit")

# Start the interactive loop
select opt in "${options[@]}"
do
    case $opt in
        "Run TecXPlayer")
            echo "--- Runnig file of  TecXPlayer---"
            cd TecXPlayer
            git pull
            clear
            python src/model/trainTecXModel.py
            ;;
        "Show System Info")
            echo "--- System Info ---"
            uname -a
            echo "--------------------"
            ;;
        "Update Packages")
            echo "Updating Termux packages..."
            pkg update && pkg upgrade -y
            ;;
        "Print Hello World")
            echo "===================="
            echo "Hello, World!"
            echo "===================="
            ;;
        "Exit")
            echo "Goodbye!"
            break
            ;;
        *) 
            echo "Invalid option $REPLY. Please choose a valid number."
            ;;
    esac
done
