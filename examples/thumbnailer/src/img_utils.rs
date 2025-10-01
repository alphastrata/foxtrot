pub fn auto_crop_image(img_data: &[u8], width: u32, height: u32, transparent: bool) -> Vec<u8> {
    // Find bounding box of non-transparent pixels
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;

    let width_usize = width as usize;
    let _height_usize = height as usize;  // Prefixed with underscore to indicate unused
    let row_size = width_usize * 4;

    for (y, row) in img_data.chunks(row_size).enumerate() {
        for (x, pixel_chunk) in row.chunks(4).enumerate() {
            let is_opaque = if transparent {
                pixel_chunk[3] > 0 // Check alpha channel
            } else {
                pixel_chunk[0] < 255 || pixel_chunk[1] < 255 || pixel_chunk[2] < 255 // Check if not white background
            };

            if is_opaque {
                min_x = min_x.min(x as u32);
                min_y = min_y.min(y as u32);
                max_x = max_x.max(x as u32);
                max_y = max_y.max(y as u32);
            }
        }
    }

    // If no content found, return original image
    if min_x >= width || min_y >= height {
        return img_data.to_vec();
    }

    let crop_width = max_x - min_x + 1;
    let crop_height = max_y - min_y + 1;
    let crop_width_usize = crop_width as usize;

    // Extract the cropped region
    let mut cropped_img_data = Vec::with_capacity((crop_width * crop_height * 4) as usize);

    (min_y..=max_y).map(|y| y as usize).for_each(|y| {
        let src_row_start = (y * width_usize + min_x as usize) * 4;
        let src_row_end = src_row_start + (crop_width_usize * 4);
        cropped_img_data.extend_from_slice(&img_data[src_row_start..src_row_end]);
    });

    cropped_img_data
}