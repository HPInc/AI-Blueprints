# OpenCellID Data Exploration with Pandas <a target="_blank" href="https://colab.research.google.com/github/AjayThorve/OpenCellID-cudf-pandas-exploration/blob/main/Exploration-Pandas.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
![demo-screenshot](./assets/demo-cell.png)

[OpenCellID](https://wiki.opencellid.org/wiki/What_is_OpenCellID) is the world's largest collaborative community project that collects GPS positions of cell towers, used free of charge, for a multitude of commercial and private purposes.

The OpenCellID project was primarily created to serve as a data source for GSM localisation. As of October, 2017, the database contained almost 36 million unique GSM Cell IDs. More than 75,000 contributors have already registered with OpenCellID, contributing millions of new measurements every day in average to the OpenCellID database.

OpenCellID provides 100% free Cell ID data under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/). The OpenCellID database is published under an open content license with the intention of promoting free use and redistribution of the data. All data uploaded by any of the contributors can also be downloaded again free of charge - no exceptions!

[![](./assets/license-opencellid.png)](https://creativecommons.org/licenses/by-sa/4.0/) [OpenCelliD Project](https://opencellid.org/) is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/)

## Dependencies

- cudf (24.04+)
- pandas (2.2.2)
- pydeck (0.9.1+)
- panel (1.4.2)
- hvplot (0.10.0+)

## Data

1. Cell-data (https://www.opencellid.org/)
   1. [Worldwide Dataset](https://data.rapids.ai/cudf/datasets/cell_towers.tar.xz) (46M rows)
   2. [US Dataset](https://data.rapids.ai/cudf/datasets/cell_towers_us.tar.xz): (7M rows) Suitable for the free tier of Google Colab.
   #### Usage in Google Colab
   - For a smaller dataset to test the notebook with the free tier of Google Colab, use: download_and_extract('us').
   #### Additional Information
   - Users can register for an account on [OpenCellID](https://www.opencellid.org/) to obtain a data access token and download the latest dataset directly.
   - The auto-downloader provided in this notebook will not fetch the latest dataset from OpenCellID. For the latest data, manual download with an access token is required.
   - If the latest data is not a priority, the included dataset dated May 2024 will suffice for exploring the notebook's functionalities.

2. MCC-MNC dataset to map to carrier name
   Publicaly available dataset: https://mcc-mnc.net/ powered by [simbase](https://www.simbase.com/)
