let colors = {};
let type_to_linetype = {'gt': [], 'pred': [5]};
let gold_colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255]];

function draw_bb(ctx, current_type_to_rects, width, height) {
    
    let placed_positions = {}
    for (let t in current_type_to_rects) {

        let rects = current_type_to_rects[t];
        for (let i = 0; i < rects.length; i++) {
            let rect_dic = rects[i];
            let rect = rect_dic['rect'];
            let label = rect_dic['class'];
            if (colors[label] == null) {
                if (gold_colors.length !== 0) {
                    colors[label] = gold_colors.pop();
                } else {
                    colors[label] = [Math.floor(Math.random() * 256), Math.floor(Math.random() * 256),
                        Math.floor(Math.random() * 256)];
                }
            }
            let label_color = colors[label];
            let color_format = 'rgb(' + label_color[0] + ','  + label_color[1] + ',' + label_color[2] + ')';

            ctx.beginPath();
            ctx.rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
            ctx.font = "20px Times";

            let text_left = [rect[0] + 2, rect[1] - 4];
            let text_bottom = [rect[0] + 2, rect[3] - 4];

            ctx.fillStyle = color_format;
            if (rect_dic['conf'] != null) {
                label = label + '[' + rect_dic['conf'].toFixed(2) + ']'
            }
            let text_x = 0;
            let text_y = 0;
            if (text_left[0] < width - 12 && text_left[0] >= 0
                && text_left[1] > 12 && text_left[1] < height) {
                text_x = text_left[0];
                text_y = text_left[1];
            } else if(text_bottom[0] < width - 12 && text_bottom[0] >= 0
                && text_bottom[1] > 12 && text_bottom[1] < height) {
                text_x = text_bottom[0];
                text_y = text_bottom[1];
            }
            while ([Math.floor(text_x), Math.floor(text_y)] in placed_positions) {
                text_y = text_y + 20;
            }
            placed_positions[[Math.floor(text_x), Math.floor(text_y)]] = 'x'
            ctx.fillText(label, text_x, text_y);

            ctx.strokeStyle = color_format;
            ctx.setLineDash(type_to_linetype[t])
            ctx.lineWidth = 3;
            ctx.closePath();
            ctx.stroke();
        }
    }
}

function draw_bb_backup(ctx, current_type_to_rects, width, height) {
    
    for (let t in current_type_to_rects) {

        let rects = current_type_to_rects[t];
        for (let i = 0; i < rects.length; i++) {
            let rect_dic = rects[i];
            let rect = rect_dic['rect'];
            let label = rect_dic['class'];
            if (colors[label] == null) {
                if (gold_colors.length !== 0) {
                    colors[label] = gold_colors.pop();
                } else {
                    colors[label] = [Math.floor(Math.random() * 256), Math.floor(Math.random() * 256),
                        Math.floor(Math.random() * 256)];
                }
            }
            let label_color = colors[label];
            let color_format = 'rgb(' + label_color[0] + ','  + label_color[1] + ',' + label_color[2] + ')';

            ctx.beginPath();
            ctx.rect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]);
            ctx.font = "20px Times";

            let text_left = [rect[0] + 2, rect[1] - 4];
            let text_bottom = [rect[0] + 2, rect[3] - 4];

            ctx.fillStyle = color_format;
            if (rect_dic['conf'] != null) {
                label = label + '[' + rect_dic['conf'].toFixed(2) + ']'
            }
            if (text_left[0] < width - 12 && text_left[0] >= 0
                && text_left[1] > 12 && text_left[1] < height) {
                ctx.fillText(label, text_left[0], text_left[1]);
            } else if(text_bottom[0] < width - 12 && text_bottom[0] >= 0
                && text_bottom[1] > 12 && text_bottom[1] < height) {
                ctx.fillText(label, text_bottom[0], text_bottom[1]);
            }

            ctx.strokeStyle = color_format;
            ctx.setLineDash(type_to_linetype[t])
            ctx.lineWidth = 3;
            ctx.closePath();
            ctx.stroke();
        }
    }
}


function make_canvas_element(canvas_obj, img_src, current_type_to_rects) {
    let ctx = canvas_obj.getContext('2d');
    let img = new Image();
    img.src = img_src;
    img.onload = function() {
        canvas_obj.width = img.width;
        canvas_obj.height = img.height;
        ctx.drawImage(img, 0, 0);
        draw_bb(ctx, current_type_to_rects, img.width, img.height);
    };
}

function draw_images(all_url, all_key) {
    let imageTable = document.getElementById('image_table');
    let num_cols = 5
    let i = 0
    while (true) {
        if (i >= all_url.length) {
            break
        }
        let row = imageTable.insertRow(-1);
        for (let c = 0; c < num_cols; c++) {
            if (i >= all_url.length) {
                break
            }
            let img_cell = row.insertCell(-1);
            let img_src = all_url[i];
            if (all_key.length > i) {
                let key = all_key[i];
                let p = document.createElement('p');
                p.appendChild(document.createTextNode(key))
                img_cell.append(p);
            } 
            let canvas = document.createElement('canvas');
            make_canvas_element(canvas, img_src, []);
            img_cell.append(canvas);
            i = i + 1
        }
    }
}

function update_label_list(type_to_rects, unchecked_labels, unchecked_type) {
    current_type_to_rects = Object.assign({}, type_to_rects)

    for (let t of unchecked_type.entries()) {
        delete current_type_to_rects[t[0]];
    }
    for (let t in current_type_to_rects) {
        let current_rects = current_type_to_rects[t];
        current_type_to_rects[t] = current_rects.filter(function(r) {return !unchecked_labels.has(r['class'])})
    }

    return current_type_to_rects;
}

function update_images(all_type_to_rects, all_url, 
    unchecked_labels, unchecked_type) {
    let unchecked_labels_set = new Set(unchecked_labels);
    let unchecked_type_set = new Set(unchecked_type)
    let imageTable = document.getElementById('image_table');

    let num_cols = 5;
    let i = 0;
    let r = 0; 
    while (true) {
        if (i >= all_type_to_rects.length) {
            break
        }
        let table_row = imageTable.rows[r];
        let row_data_list = table_row.cells;

        for (let c = 0; c < num_cols; c++) {
            if (i >= all_type_to_rects.length) {
                break
            }
            let type_to_rects = all_type_to_rects[i];
            let img_src = all_url[i];
            i = i + 1

            let all_l_table_data = row_data_list[c];  // all label image
            let canvas = all_l_table_data.getElementsByTagName('canvas')[0];
            let all_label_info = type_to_rects['all_info'];
            current_type_to_rects = update_label_list(type_to_rects, unchecked_labels_set,
                unchecked_type_set);
            make_canvas_element(canvas, img_src, current_type_to_rects);
        }
        r = r + 1;
    }

}

function get_unchecked_list(name) {
    let label_ul = document.getElementById(name);
    let inputs = label_ul.getElementsByTagName('input')
    let unchecked = [];
    for (let i = 0; i < inputs.length; i++) {
        if (!inputs[i].checked) {
            unchecked.push(inputs[i].id)
        }
    }
    return unchecked;
}
function add_type_points(type_list) {
    let label_table = document.getElementById("type_table");

    let row = label_table.insertRow(-1)
    for (let i = 0; i < type_list.length; i++) {

        let label_checkbox = document.createElement('input');
        label_checkbox.type = "checkbox";
        label_checkbox.id = type_list[i];
        label_checkbox.onclick = onClick_abstraction;
        label_checkbox.checked = true;

        let label = document.createElement('label');
        label.htmlFor = type_list[i];
        label.appendChild(document.createTextNode(type_list[i]));
        
        let cell = row.insertCell(-1)
        cell.appendChild(label_checkbox);
        cell.appendChild(label);
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


