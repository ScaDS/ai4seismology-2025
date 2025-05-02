# Setting-up

## Access to the HPC cluster

During the training school, you will have access to the HPC infrastructure of TU Dresden to run the tutorial notebooks. To connect, you will need to set-up the TU Dresden [VPN connection](https://tu-dresden.de/zih/dienste/service-katalog/arbeitsumgebung/zugang_datennetz/vpn?set_language=en). It is recommended to install the [eduVPN](https://tu-dresden.de/zih/dienste/service-katalog/arbeitsumgebung/zugang_datennetz/vpn?set_language=en) client, which is available for all major operating systems.

To set-up eduVPN, [download the client](https://www.eduvpn.org/client-apps/) for your operating system and follow the instructions provided. After the installation completes, open eduVPN, enter "Dresden" in the search bar, and click "TUD Dresden University of Technology". A browser window will open that prompts you for a username and password. These are the credentials that have been given to you on the first day of the school. After successful login, you will be asked to confirm access for the eduVPN client (click the red button).

Once this is done, you can return to the eduVPN client and select "TUD VPN full" to route all traffic through the VPN. It may take a few seconds to establish the connection.

The HPC cluster is now accessible from: https://jupyterhub.hpc.tu-dresden.de (provide the same login credentials as for the VPN)

## Setting-up your workspace

1. Go to https://jupyterhub.hpc.tu-dresden.de and log in
2. Click the "Advanced" button
3. Select the cluster "barnard" and set "CPUs per task", "Memory per CPU", and "Runtime" according to the tutorial instructions
4. After being assigned a spot on the cluster, you will be greeted by a Jupyter Lab environment. Open a terminal and copy the tutorial environment into your home directory: `cp -r /data/horse/ws/s4122485-ai4seismology .` 
5. Close the terminal. In the file browser on the left side, click the refresh icon and open `s4122485-ai4seismology_dev/HPCSetup.ipynb` 
6. When asked to "Select Kernel", leave the default Python 3 kernel and click "Select".
7. Run the first cell in the notebook to install the `ai4seis_kernel`, which contains the Python environment needed to run the tutorial notebooks. If everything went well (no errors), you can close the `HPCSetup.ipynb` notebook.

Steps 4+ are only required once. For subsequent tutorials, you follow steps 1-3 and continue where you left off.
