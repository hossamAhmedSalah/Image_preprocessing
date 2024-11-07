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

# Histogram 

- Information contained in the graph is a representation of pixel distribution as a function of tonal variation
- A histogram uses a bar graph to profile the occurrences of each gray level present in an image.
- The horizontal axis is the gray-level values.
- It begins at zero and goes to the number of gray levels (256 in this example).
- Each vertical bar represents the number of times the corresponding gray level occurred in the image.
- Histograms also help select thresholds for object detection
  - Objects in an image tend to have similar gray levels.
  
# Histogram Equalization

- Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.
- This method usually increases the global contrast of many images, especially when the image is represented by a narrow range of intensity values.
- Through this adjustment, the intensities can be better distributed on the histogram utilizing the full range of intensities evenly. 
- This allows for areas of lower local contrast to gain a higher contrast.
- Histogram equalization accomplishes this by effectively spreading out the highly populated intensity values which are used to degrade image contrast.
- It is a Straightforward technique adaptive to the input image and an invertible operator.
  - if the histogram equalization function is known, then the original histogram can be recovered.
- The method is indiscriminate
  - It may increase the contrast of background noise
  - while decreasing the usable signal.
- Histogram equalization will work the best when applied to images with much higher color depth than palette size, like continuous data or 16-bit gray-scale images.

![image.png](attachment:image.png)

## Equalization Algorithm 
1. Calculate the histogram of the image 
   - histogram is just the frequency/count of each pixel values in the image.
   - if the image has $N×M$ pixels.
   - for each pixel we would calculate the number of occurence $n_i$ which is the histogram $H(i)$
   - the probability of each intensity value $i$ in the image is $p(i)$ 
   $$
   \huge p(i) = \frac{n_i}{Area} = \frac{H(i)}{N×M} 
   $$
2. Compute the cumulative distribution function (CDF)
   - The CDF is used to map the original intensities to the new, equalized intensities.
   - compute the cumulative sum for each intensity level $i$
   $$
   \huge C(i) = \sum_{j=0}^{i} p(j)
   $$
   - $C(i)$ gives the cumulative probability up to intensity $i$.
   - This step effectively "spreads" the intensity values over the whole range.
   - $C(i)$ ranges from 0 and 1.
3. Map Each Pixel to the New Intensity Value
   - Using the CDF, we can map each original intensity value to a new one that distributes the pixel intensities more evenly across the histogram.
   - multiply each $C(i)$ by the maximum intensity value (255 for 8-bit images)
   $$

   \huge new-intensity(i) = int(255*C(i))
   $$
![alt text](image-2.png)