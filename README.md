# Image_preprocessing
A series of notebooks and notes about the image processing course in Helwan university 
# RGB to gray scale 
- R 30%, G 60%, B 10%
$$
\huge I = \red{R * 0.2989} + \green{G *  0.5870} + \blue{B * 0.1140} 
$$

- **Green has the highest weight **
- **Blue has the lowest weight**
  
## other methods to convert RGB image to gray scale 
$$
\huge I = \red{R * 0.2989} + \green{G *  0.5870} + \blue{B * 0.1140} 
$$

| Method                  | Formula                                      | Description                                                                                           | Pros                           | Cons                                 |
|-------------------------|----------------------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------|--------------------------------------|
| **Luminosity Method**   | $0.2989 * R + 0.5870 * G + 0.1140 * B$       | Uses weighted average based on human perception of color brightness.                                  | More realistic grayscale       | Slightly more complex calculation    |
| **Average Method**      | $\large \frac{(R + G + B) }{3}$                            | Averages all three color channels equally.                                                            | Simple and fast                | Ignores color perception, less realistic |
| **Desaturation Method** | $\large \frac{(max(R, G, B) + min(R, G, B))} {2}$         | Averages the lightest and darkest colors in each pixel.                                               | Higher contrast                | Can lose detail                      |
| **Single Channel**      | `R` (or `G` or `B` channel)                  | Uses only one color channel as the grayscale intensity.                                               | Very simple and fast           | Ignores information from other channels |
| **ITU-R BT.601**        |$0.299 * R + 0.587 * G + 0.114 * B  $        | Similar to luminosity but defined by the ITU-R BT.601 standard, often used in video processing.       | High accuracy for video        | Slightly more complex calculation    |
```python
# function that turn the image to gray scale 
def RGB_to_gray(image):
    # we can use this but some images may have a fourth channel 
    # return np.dot(img, [0.2989, 0.5870, 0.1140])
    # but using img[..., :3] or img[:, :, :3] is taking care of this possibility 

    return np.dot(img[:, :, :3], [0.2989, 0.5870, 0.1140])
plt.imshow(RGB_to_gray(img), cmap='gray');
```

# Halftoning 
## Simple Halftoning 
```python
def halftoning_elite(img, t=128):
    if len(img.shape) > 2 : return
    # if a pixel intensity higher than t
    # assign to it 255 else 0 (white/black)
    return np.where(img >= t, 255, 0)
plt.imshow(halftoning(gimg), cmap='gray');
```
## Advanced Halftoning with Error diffusion 

![alt text](image.png)

```python
# original gray scale image -> gimg
# halftoned image -> himg 
# Error at given pixel is old_pixel_value - new_pixel_value 
err_diff_img = np.copy(himg)

for ir, R in enumerate(himg):
    for ic, C in enumerate(R):
        # Error = old - new
        #  THIS IS THE STANDARD WAY AS WE CACULATING-THE-ERROR-ON-THE-SAME-IMAGE-WE-WOULD-RETURN NOT ON UNCHANGED IMAGE
        err = gimg[ir, ic] - err_diff_img[ir, ic]
        # let's propagate the error 
        bot_err, bot_lef_err, bot_rig_err, rig_err = ((1/16)*err)*np.array([5, 3, 1, 7])
        try:
            # right 
            err_diff_img[ir, ic+1] += rig_err
            # bottom 
            err_diff_img[ir+1, ic] += bot_err 
            err_diff_img[ir+1, ic+1] += bot_rig_err
            err_diff_img[ir+1, ic-1] += bot_lef_err
        except:
            continue
# Clip values to stay within valid range [0, 255]
err_diff_img = np.clip(err_diff_img, 0, 255)

plt.imshow(err_diff_img, cmap='gray')
plt.axis('off')
plt.show()
```
with padding to simplify the code 
```python 
# optimized but more simple way is to just add padding 
err_diff_img = np.copy(himg)
# padding
err_diff_img_pad = np.pad(err_diff_img, pad_width=1, constant_values=0)
for ir in range(err_diff_img.shape[0]):
    for ic in range(err_diff_img.shape[1]):
        err = gimg[ir, ic] - err_diff_img_pad[ir, ic]
        err_diff_img_pad[ir, ic+1] += err*(7/16)
        err_diff_img_pad[ir+1, ic] += err*(5/16)
        err_diff_img_pad[ir+1, ic-1] += err*(3/16)
        err_diff_img_pad[ir+1, ic+1] += err*(1/16)

un_padded_img = err_diff_img_pad[1:-1, 1:-1]
un_padded_img = np.clip(un_padded_img, 0, 255)
plt.imshow(un_padded_img, cmap='gray')
plt.axis('off')
plt.show()

```
![alt text](image-1.png)

