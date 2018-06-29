input_dir = 'CarND-LaneLines-P1/test_images/'
output_dir = 'CarND-LaneLines-P1/test_images_output/'

canny_low = 50
canny_high = 150

region_ratios = [
    (0, 1),
    (0.45, 0.60),
    (0.55, 0.60),
    (1, 1)
]

hough_rho = 2
hough_theta = math.pi/180
hough_threshold = 15
hough_min_line_len = 40
hough_max_line_gap = 20

os.makedirs(output_dir, exist_ok=True)

for image_name in os.listdir(input_dir):
    print(image_name)
    image = mpimg.imread(os.path.join(input_dir, image_name))
    
    img_width = image.shape[1]
    img_height = image.shape[0]
    region_points = [(int(ratio[0] * img_width), int(ratio[1] * img_height)) for ratio in region_ratios]

    vertices = np.array([region_points], dtype=np.int32)
    
    gray_image = grayscale(image)
    blur_image = gaussian_blur(gray_image, 5)
    edge_image = canny(blur_image, canny_low, canny_high)
    masked_edge_image = region_of_interest(edge_image, vertices)
    lines = hough_lines(masked_edge_image, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)
    result = weighted_img(lines, image)
    
    mpimg.imsave(os.path.join(output_dir, image_name), result, None, None, None, 'jpg')
    plt.figure()
    plt.imshow(result)
