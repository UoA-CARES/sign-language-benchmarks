 #!/bin/bash 
 
cd ../..

# Change the external drive name here
EXTERNALDRIVE='Sadat'

# Set up the external drive with the following structure
: '
EXTERNALDRIVE
├── autsl
│   └── data
└── ...
 '
mkdir -p "/media/${USER}/${EXTERNALDRIVE}/autsl"
mkdir -p "/media/${USER}/${EXTERNALDRIVE}/autsl/data"

# Create the symbolic link
ln -s "/media/${USER}/${EXTERNALDRIVE}/autsl/data" data


