import subprocess
import sys

def move_mouse(x, y):
  subprocess.Popen(['powershell.exe',
  '-ExecutionPolicy',
  'Unrestricted',
  'C:\\Users\\GZhang\\Desktop\\side-projects\\machine-learning\\siege_assist\\mouse_mover.ps1',
  '-x', str(x),
    '-y', str(y)])