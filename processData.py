import re


def process_chat_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    processed_lines = []
    current_speaker = None
    current_message = []

    for line in lines:
        # Remove timestamp but keep the date
        date_match = re.match(r"^(\d{2}/\d{2}/\d{4}), \d{1,2}:\d{2}\s?[ap]m - ", line)
        if date_match:
            current_date = date_match.group(1)
            line = re.sub(
                r"^\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?[ap]m - ",
                f"{current_date} - ",
                line,
            ).strip()

        # Skip media omitted lines
        if "<Media omitted>" in line:
            continue

        if ": " in line:
            speaker, message = line.split(": ", 1)
            if speaker == current_speaker:
                current_message.append(message)
            else:
                if current_speaker is not None:
                    processed_lines.append(
                        f"{current_speaker}: {' * '.join(current_message)}"
                    )
                current_speaker = speaker
                current_message = [message]
        else:
            if current_speaker is not None:
                current_message.append(line)

    if current_speaker is not None:
        processed_lines.append(f"{current_speaker}: {' * '.join(current_message)}")

    with open(output_file, "w", encoding="utf-8") as file:
        for line in processed_lines:
            file.write(line + "\n")


# Usage
input_file = "data/WhatsApp Chat with Prathwik.txt"
output_file = "processed_chat.txt"
process_chat_file(input_file, output_file)
