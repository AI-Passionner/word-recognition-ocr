class TextractUtils:
    """need to change"""
    def __init__(self, textract_response):
        """
        textract_response: Textract   response    files    JSON    representation
        """

        self.response = textract_response
        self.Text = ''
        self.word_map = {}
        self.line_map = {}

    def TextractParser(self):
        """
        Parses the Textract JSON Object and returns the word map and line map
        """

        blocks = self.response['Blocks']
        # get  word maps       
        for block in blocks:
            block_id = block['Id']
            if block['BlockType'] == "WORD":
                left = block['Geometry']['BoundingBox']['Left']
                top = block['Geometry']['BoundingBox']['Top']
                height = block['Geometry']['BoundingBox']['Height']
                width = block['Geometry']['BoundingBox']['Width']
                right = left + width
                bottom = top + height
                self.word_map[block_id] = {'ids': [block_id],
                                           'Text': block['Text'].replace(' ', ''),
                                           'text': block['Text'].replace(' ', '').lower(),
                                           'left': left,
                                           'right': right,
                                           'top': top,
                                           'bottom': bottom,
                                           'width': width,
                                           'height': height,
                                           'confidence': block['Confidence']
                                           }

            elif block['BlockType'] == "LINE":
                if 'Relationships' in block.keys():
                    for relationship in block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            ids = relationship['Ids']

                    left = block['Geometry']['BoundingBox']['Left']
                    top = block['Geometry']['BoundingBox']['Top']
                    height = block['Geometry']['BoundingBox']['Height']
                    width = block['Geometry']['BoundingBox']['Width']
                    right = left + width
                    bottom = top + height
                    self.line_map[block_id] = {'ids': ids,
                                               'Text': block['Text'].replace(' ', ''),
                                               'text': block['Text'].replace(' ', '').lower(),
                                               'left': left,
                                               'right': right,
                                               'top': top,
                                               'bottom': bottom,
                                               'width': width,
                                               'height': height,
                                               'confidence': block['Confidence']
                                               }
        return self.word_map, self.line_map

    def GetText(self):
        """
        return a full text of the document
        """

        for item in self.response["Blocks"]:
            if item["BlockType"] == "WORD":
                self.Text = self.Text + " " + item["Text"]

        self.Text = self.Text.strip()

        return self.Text
