
rm a.json
git pull
# clear
python src/model/trainModel.py
pause
# A clean, simple pause with a custom prompt message
read -p "Press [Enter] key to continue..."

exit
# Force kills the entire running script immediately
kill -9 $$

# Silent pause (waits for a single keypress without requiring Enter)
read -n 1 -s -r -p "Press any key to continue..."

run.sh
pause
echo " I am exiting!"
exit
#
exit 1
#
#
#!/bin/bash

echo "Running batch step..."

# Clear fix for the Windows 'pause' error
read -p "Script paused. Press Enter to exit..."

# Clear fix for a stuck loop or unresponsive exit command
kill -9 $$

#
#

chmod +x TecXPlayer/run.sh
./TecXPlayer/run.sh
pause
cd TecXPlayer
./chmod +x run.sh
run.sh
pause
cd ~
chmod +x ./TecXPlayer/run.sh
cd TecXPlayer
./run.sh
pause
#
nano ~/.bashrc
# Automatically run my batch process on startup
bash ~/TP_run.sh

exit

chmod +x ~/TP_run.sh
