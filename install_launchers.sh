#!/bin/bash
# install_launchers.sh -- puts two launchers on the Raspberry Pi desktop:
#   "Update Endoscope"  pulls the latest software from GitHub (shows output)
#   "Endoscope"         starts the camera viewer (errors pop up in a window)
# Run once:  cd ~/Pidev && bash install_launchers.sh
set -e
REPO="$HOME/Pidev"
DESK="$HOME/Desktop"
APPS="$HOME/.local/share/applications"
mkdir -p "$DESK" "$APPS" "$HOME/.config/libfm"

TERM_CMD=lxterminal
command -v lxterminal >/dev/null 2>&1 || TERM_CMD=x-terminal-emulator

# --- 1. update script: pull, show the result, wait for Enter
cat > "$REPO/update.sh" <<'EOF'
#!/bin/bash
cd "$HOME/Pidev" || { echo "Pidev folder not found"; read -p "Press Enter"; exit 1; }
echo "Updating endoscope software from GitHub ..."
echo
git pull
echo
echo "---- Done. If it says 'Already up to date', GitHub has nothing newer."
read -p "Press Enter to close this window"
EOF
chmod +x "$REPO/update.sh"

# --- 2. run script: start the viewer; if it crashes, show the log
cat > "$REPO/run.sh" <<EOF
#!/bin/bash
cd "\$HOME/Pidev"
export DISPLAY="\${DISPLAY:-:0}"
LOG="\$HOME/endoscope.log"
: > "\$LOG"
python3 endoscope.py >> "\$LOG" 2>&1 || $TERM_CMD -e bash -c "echo 'Endoscope exited with an error:'; echo; tail -n 40 '\$LOG'; echo; read -p 'Press Enter to close'"
EOF
chmod +x "$REPO/run.sh"

# --- 3. desktop entries
cat > "$DESK/Update Endoscope.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Update Endoscope
Comment=Download the latest endoscope software from GitHub
Exec=$TERM_CMD -e bash $REPO/update.sh
Icon=system-software-update
Terminal=false
Categories=Utility;
EOF

cat > "$DESK/Endoscope.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Endoscope
Comment=Start the endoscope camera viewer
Exec=bash $REPO/run.sh
Icon=camera-video
Terminal=false
Categories=Utility;
EOF
chmod +x "$DESK/Update Endoscope.desktop" "$DESK/Endoscope.desktop"
cp "$DESK/Update Endoscope.desktop" "$DESK/Endoscope.desktop" "$APPS/"

# --- 4. let a click run the launcher without the "Execute?" question
CONF="$HOME/.config/libfm/libfm.conf"
if [ -f "$CONF" ] && grep -q '^quick_exec=' "$CONF"; then
  sed -i 's/^quick_exec=.*/quick_exec=1/' "$CONF"
elif [ -f "$CONF" ] && grep -q '^\[config\]' "$CONF"; then
  sed -i '/^\[config\]/a quick_exec=1' "$CONF"
else
  printf '[config]\nquick_exec=1\n' >> "$CONF"
fi

echo
echo "OK: two launchers are now on the desktop:  'Update Endoscope'  and  'Endoscope'"
echo "    (also in the application menu under Accessories/Utility)"
