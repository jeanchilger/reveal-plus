import xml.etree.ElementTree as ET

tree = ET.parse("topics-rnd1.xml")
root = tree.getroot()

with open("cord-19.topic.stemming.txt", "a") as cord_base_file:
    for elem in root:
        query = elem[0].text
        topic = "tr" + elem.attrib["number"]

        cord_base_file.write(topic + ":" + query + "\n")
