# Operon's cloud computer (Azure Linux VM, KAN-108)

Operon can run on its own computer in the cloud instead of yours: one small Debian 12
VM with an XFCE desktop. It runs only while a task does. You watch the task live on the
task page (`/app`), and can take over and drive it from there.

```
your PC                                    Azure VM (westus2, rg operon-bench)
operon cloud open/run  --az vm start-->    Xvfb :1 + XFCE desktop
        |                                  Operon API on 127.0.0.1:8080
        +--- SSH tunnel 127.0.0.1:8090 --> (OPERON_DESKTOP_PLATFORM=linux)
browser: http://127.0.0.1:8090/app#key=...
```

- **Nothing on the VM is public except SSH.** SSH is open only to one IP (yours, when you
  create the VM). The task page and its API listen on the VM's localhost, and you reach them
  through the tunnel.
- **Billing.** About $0.10 an hour while the VM runs (Standard_D2s_v4, 2 vCPU, 8 GB, westus2
  list price). The disk ($2.40 a month, 30 GB Standard SSD) and the public IP ($3.65 a month)
  are kept between tasks. `open` and `run` deallocate the VM when they end, even after an
  error or Ctrl+C. Azure also auto-deallocates it every day at 08:00 UTC, in case a session
  was left open.
- **Keys.** `operon cloud set-key` sends your `GEMINI_API_KEY` to the VM over SSH's stdin,
  into `/etc/operon/gemini.env` (root only). The task page's own API key is generated on the
  VM, and `operon cloud` reads it over SSH.
  Apps that Operon launches don't inherit secret environment variables. But they run as the
  same user as the API, so treat the VM as a sandbox: don't sign it in to anything you care about.

## Commands (from your PC; needs `az login` and `ssh`)

| Command | What it does |
|---|---|
| `operon cloud create [--ref BRANCH]` | Creates the VM once (about 10 minutes of setup). Allows SSH from this PC's IP only and sets the daily auto-deallocate. |
| `operon cloud set-key` | Saves `GEMINI_API_KEY` on the VM, then deallocates it. |
| `operon cloud open` | Starts the VM, opens the tunnel and prints the task page link. Pick **Desktop** on the page. Ctrl+C deallocates. |
| `operon cloud run "task"` | Starts the VM, runs one desktop task (link printed so you can watch), then deallocates. Add `--browser` to use the VM's browser instead. |
| `operon cloud stop` | Deallocates the VM now. |

All of them accept `--keep` to leave the VM running. Settings: `OPERON_CLOUD_*` in `.env.example`.

## The Linux desktop

`OPERON_DESKTOP_PLATFORM=linux` switches the desktop loop to `LinuxDesktopExecutor`
(`src/operon/executor/linux_desktop.py`). It uses the same pyautogui/mss executor, with these differences:

- `launch_app` uses `LINUX_APP_LAUNCH_TARGETS` (Mousepad, Galculator, Thunar, Chromium,
  Firefox ESR, LibreOffice Calc and Writer, and others). Apps start as argv lists with no shell,
  each in its own process group, so cleanup ends the app and its children.
- The prompt is `desktop_computer_use_prompt_linux.txt`.
- The task page gets JPEG frames of the screen over `/ws/stream` (4 a second, at most 1280 px
  wide), and during a takeover it sends your clicks, keys and scrolls to the VM.

A local Windows desktop is unchanged. Its runs still show step screenshots, and you take over
with your own mouse and keyboard.

## If something goes wrong

- Setup log: `ssh azureuser@<ip> sudo tail -50 /var/log/cloud-init-output.log`
- API log: `ssh azureuser@<ip> sudo journalctl -u operon-api -n 50`
- Your IP changed (SSH times out): `az network nsg rule update -g operon-bench --nsg-name operon-linuxNSG -n ssh-from-owner --source-address-prefixes <new-ip>/32`
- Update Operon on the VM: `ssh azureuser@<ip> "sudo -u operon git -C /home/operon/Operon pull && sudo systemctl restart operon-api"`
