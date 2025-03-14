# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]



dev:
    cd frontend/react-web | pnpm run dev

clear:
    rd /s /q node_modules | del package-lock.json

install:
    cd frontend/react-web | pnpm install

clear-install:
  just clear
  just install