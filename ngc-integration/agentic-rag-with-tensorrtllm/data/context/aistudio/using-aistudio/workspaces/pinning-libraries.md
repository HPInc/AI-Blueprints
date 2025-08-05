---
title:  'Pinning Libraries'
sidebar_position: 2
---

# Pinning Libraries

When cloning projects, ensure you clone them locally to a repository or folder. For instance, AI Studio projects should be cloned to a designated location. Within these projects, you may find a requirements.txt file.

Creating a New Project in AI Studio
1.	Navigate to Workspace: Go to your workspace in AI Studio.
2.	Upload Requirements File: Upload the requirements.txt file associated with your project.


## Managing Library Dependencies

When specifying library versions in the requirements.txt file, use the format library==version. Be aware that pinning libraries to specific versions can sometimes cause dependency issues with the container image.

:::tip **Recommendations**:

- **Verify Compatibility**: Ensure that pinned libraries are compatible with your project.
- **Adjust Versions if Necessary**: If compatibility issues arise, remove the ==version specification for the libraries and update them accordingly.

:::
