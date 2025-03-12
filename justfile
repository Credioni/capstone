# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]

clear:
    rd /s /q node_modules && del package-lock.json

front build:
    npm