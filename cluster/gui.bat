@echo off
REM Launcher for cluster\gui.pyw -- double-click this if Windows didn't
REM associate .pyw files with pythonw.exe automatically. The GUI runs
REM detached from this console window so you can close the cmd window
REM after it pops up if you don't want to see future stderr lines.

start "" pythonw.exe "%~dp0gui.pyw"
