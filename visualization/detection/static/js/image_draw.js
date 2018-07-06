function draw_bb(ctx, rects, labels, width, height) {
    let gold_colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]];
    let colors = {};

    for (let i = 0; i < labels.length; i++) {
        let l = labels[i];
        if (colors[l] == null) {
            if (gold_colors.length !== 0) {
                colors[l] = gold_colors.pop();
            } else {
                colors[l] = [Math.floor(Math.random() * 256), Math.floor(Math.random() * 256),
                    Math.floor(Math.random() * 256)];
            }
        }
    }

    for (let i = 0; i < rects.length; i++) {
        let rect = rects[i];
        let label = labels[i];
        let label_color = colors[label];
        let color_format = 'rgb(' + label_color[0] + ','  + label_color[1] + ',' + label_color[2] + ')';

        ctx.beginPath();
        ctx.rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
        ctx.font = "20px Times";

        let text_left = [rect[0] + 2, rect[1] - 4];
        let text_bottom = [rect[0] + 2, rect[3] - 4];

        ctx.fillStyle = color_format;
        if (text_left[0] < width - 12 && text_left[0] >= 0
            && text_left[1] > 12 && text_left[1] < height) {
            ctx.fillText(label, text_left[0], text_left[1]);
        } else if(text_bottom[0] < width - 12 && text_bottom[0] >= 0
            && text_bottom[1] > 12 && text_bottom[1] < height) {
            ctx.fillText(label, text_bottom[0], text_bottom[1]);
        }

        ctx.strokeStyle = color_format;
        ctx.lineWidth = 3;
        ctx.closePath();
        ctx.stroke();
    }
}


function make_canvas_element(canvas_obj, img_src, label_info_rect, label_info_class) {
    let ctx = canvas_obj.getContext('2d');
    let img = new Image();
    img.src = img_src;
    img.onload = function() {
        canvas_obj.width = img.width;
        canvas_obj.height = img.height;
        ctx.drawImage(img, 0, 0);
        draw_bb(ctx, label_info_rect, label_info_class, img.width, img.height);
    };
}


function draw_images(images, static_path, label_list) {
    let imageTable = document.getElementById('image_table');
    let num_cols = 5
    let i = 0
    while (true) {
        if (i >= images.length) {
            break
        }
        let row = imageTable.insertRow(-1);
        for (let c = 0; c < num_cols; c++) {
            if (i >= images.length) {
                break
            }
            let image_data = images[i];
            i = i + 1
            let img_src = static_path + image_data['path'];

            let img_cell = row.insertCell(-1);
            let canvas = document.createElement('canvas');
            make_canvas_element(canvas, img_src, [], []);
            img_cell.append(canvas);
        }
    }
    add_label_points(label_list)
}


function update_label_list(label_info, unchecked_set) {
    class_list = label_info['class'].slice();
    rects_list = label_info['rect'].slice();

    for (let j = 0; j < label_info['class'].length; j++) {
        let label = label_info['class'][j];
        if (!unchecked_set.has(label)) { continue }

        let sliceIndex = class_list.indexOf(label);
        if (sliceIndex >= 0) {
            class_list.splice(sliceIndex, 1);
            rects_list.splice(sliceIndex, 1);
        }
    }
    return [class_list, rects_list];
}


function update_images(images, static_path, unchecked_labels) {
    let unchecked_labels_set = new Set(unchecked_labels);
    let imageTable = document.getElementById('image_table');

    let num_cols = 5;
    let i = 0;
    let r = 0; 
    while (true) {
        if (i >= images.length) {
            break
        }
        let table_row = imageTable.rows[r];
        let row_data_list = table_row.cells;

        for (let c = 0; c < num_cols; c++) {
            if (i >= images.length) {
                break
            }
            let image_data = images[i];
            i = i + 1
            let img_src = static_path + image_data['path'];

            let all_l_table_data = row_data_list[c];  // all label image
            let all_l_canvas = all_l_table_data.getElementsByTagName('canvas')[0];
            let all_label_info = image_data['all_info'];
            [all_label_info_class, all_label_info_rect] = update_label_list(all_label_info, unchecked_labels_set);
            make_canvas_element(all_l_canvas, img_src, all_label_info_rect, all_label_info_class);
        }
        r = r + 1;
    }

}

function add_label_points(label_list) {
    let label_table = document.getElementById("label_table");

    let row = label_table.insertRow(-1)
    for (let i = 0; i < label_list.length; i++) {

        let label_checkbox = document.createElement('input');
        label_checkbox.type = "checkbox";
        label_checkbox.id = label_list[i];
        label_checkbox.onclick = onClick_abstraction;
        label_checkbox.checked = i == 0;

        let label = document.createElement('label');
        label.htmlFor = label_list[i];
        label.appendChild(document.createTextNode(label_list[i]));
        
        let cell = row.insertCell(-1)
        cell.appendChild(label_checkbox);
        cell.appendChild(label);
    }
}

