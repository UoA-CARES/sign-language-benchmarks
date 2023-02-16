 #!/bin/bash 
 
cd ../..

# Change the external drive name here
EXTERNALDRIVE='Sadat'

# Set up the external drive with the following structure
: '
EXTERNALDRIVE
├── wlasl
│   └── data
└── ...
 '
mkdir -p "/media/${USER}/${EXTERNALDRIVE}/wlasl"
mkdir -p "/media/${USER}/${EXTERNALDRIVE}/wlasl/data"

# Create the symbolic link
ln -s "/media/${USER}/${EXTERNALDRIVE}/wlasl/data" data


