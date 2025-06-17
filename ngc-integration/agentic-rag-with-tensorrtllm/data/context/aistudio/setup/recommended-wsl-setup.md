---
title:  'Recommended WSL Setup'
sidebar_position: 3
---
# Recommended WSL Setup

Follow these instructions to configure WSL and prepare your Windows machine for AI Studio.

### To configure Windows for WSL:  

1. From the Start menu, search for ***Turn Windows features on or off*** and select it.

2. Enable the following features:
   
	- Virtual Machine Platform

	- Windows Subsystem for Linux 

3. If prompted, restart your computer to apply changes.

### To set up WSL:

1. From the Start menu, search for ***Windows PowerShell***, right-click it, and select **Run as administrator**.

2. Run the following command:

	```powershell
		wsl --install --no-distribution
	```

3. Restart your computer to apply changes.

## Troubleshooting

### WSL is installed but it doesn't work or AI Studio doesn't recognize it.

Sometimes a required Windows feature might not be enabled properly. We can turn the feature off and back on to try and fix it.

1. From the Start menu, search for ***Turn Windows features on or off*** and select it.

2. Disable the following features:
   
	- Virtual Machine Platform

	- Windows Subsystem for Linux
3. If prompted, restart your computer to apply changes.

4. Open ***Turn Windows features on or off*** again.

5. Re-enable the following features:
   
	- Virtual Machine Platform

	- Windows Subsystem for Linux

6. If prompted, restart your computer again to apply changes.

7. Try using AI Studio again.

### Other WSL problems

Try to follow the instructions above for setting up WSL, even if it is already set up.

You can also try this command:

```powershell
wsl --update
```

If you're still experiencing issues, click here for help and support: [Get Help](/get-help).
