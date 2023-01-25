s-Line-22-CO_1+D_cut.uvt
    read uv s-Line-22-CO_1+D_cut.uvt /range 0 72 channel
s-Line-18-13CO_4+D_cut.uvfits
    /range 9 55 channel
s-Line-29-HCO+_1+D_cut.uvt
    /range 0 74

fits "" to "name"  # to open uvftis files and define its uv data in the variable "name"
write uvt ""  # to save the uvt in the buffer
fits image.uvfits from imaged.uvt /style casa