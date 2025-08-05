# Windows WSL Memory Management Advisory

## Issue: High Memory Consumption in Windows WSL

When running AIS on the Windows platform through WSL (Windows Subsystem for Linux), users frequently experience high memory utilization (often reaching 80% or more). This issue is specific to Windows WSL and doesn't affect native Linux installations.

## Technical Explanation

The issue occurs because:

1. Linux kernel allocates memory pages for I/O operations (file I/O, network I/O) as buffers
2. Linux filesystems (Ext3/4) allocate memory pages for cache to improve performance
3. When pulling images or running workspaces in WSL, the Linux kernel allocates large amounts of memory for caching
4. MS Hyper-V treats this cache memory as actual required memory and allocates Windows host memory accordingly
5. This results in excessive Windows memory consumption

## Solution: Clearing Linux Cache Memory

You can significantly reduce Windows memory usage by instructing the Linux kernel to drop cached memory.

Open a CMD prompt and run the following command:
```bash
wsl.exe -d phoenix
```

Then run this command:
```bash
echo 3 > /proc/sys/vm/drop_caches
```
This signals the kernel to release non-essential memory allocations, including filesystem caches, allowing the Windows WSL VM to deallocate host memory.

We recommend running this each time after you stop an AIS Workspace if you are concerned about the memory usage. We intend to apply an intelligent, programmatic form of this fix to AIS in a near term app release.
