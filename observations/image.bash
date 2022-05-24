rm data.imager
printf "LUT inferno\nLET SIZE 10\nLET DO_CONTOUR NO\n" >> data.imager
for FILE in *_cut.uvt; do 
    printf "READ UV ${FILE%.*}\nUV_MAP\nCLEAN\nSHOW\nHARDCOPY ${FILE%.*}.png /DEVICE png /OVERWRITE\n" >> data.imager; 
done
imager -nw < data.imager
