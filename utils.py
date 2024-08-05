import re


def get_text_inside_tag(html_string: str, tag: str):
    # html_string 为待解析文本，tag为查找标签
    pattern = f"<{tag}>(.*?)<\/{tag}>"
    try:
        result = re.findall(pattern, html_string, re.DOTALL)
        return result
    except SyntaxError as e:
        raise ("Json Decode Error: {error}".format(error=e))


def save_triplets_to_txt(triplets, file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{triplets[0]},{triplets[1]},{triplets[2]}\n")
