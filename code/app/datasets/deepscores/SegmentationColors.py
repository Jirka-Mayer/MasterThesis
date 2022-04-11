from PIL import ImageColor


class SegmentationColors:
    # Colors used in deepscores v2 for semantic segmentation.
    # Extracted by installing the "colorcet" package and asking for "colorcet.glasbey".
    # Obtained from the code at:
    # https://github.com/yvan674/obb_anns/blob/master/obb_anns/obb_anns.py#L791
    COLORCET_GLASBEY = [
        '#d60000', '#8c3bff', '#018700', '#00acc6', '#97ff00', '#ff7ed1', '#6b004f',
        '#ffa52f', '#573b00', '#005659', '#0000dd', '#00fdcf', '#a17569', '#bcb6ff',
        '#95b577', '#bf03b8', '#645474', '#790000', '#0774d8', '#fdf490', '#004b00',
        '#8e7900', '#ff7266', '#edb8b8', '#5d7e66', '#9ae4ff', '#eb0077', '#a57bb8',
        '#5900a3', '#03c600', '#9e4b00', '#9c3b4f', '#cac300', '#708297', '#00af89',
        '#8287ff', '#5d363b', '#380000', '#fdbfff', '#bde6bf', '#db6d01', '#93b8b5',
        '#e452ff', '#2f5282', '#c36690', '#54621f', '#c49e72', '#038287', '#69e680',
        '#802690', '#6db3ff', '#4d33ff', '#85a301', '#fd03ca', '#c1a5c4', '#c45646',
        '#75573d', '#016742', '#00d6d4', '#dadfff', '#f9ff00', '#6967af', '#c39700',
        '#e1cd9c', '#da95ff', '#ba03fd', '#915282', '#a00072', '#569a54', '#d38c8e',
        '#364426', '#97a5c3', '#8e8c5e', '#ff4600', '#c8fff9', '#ae6dff', '#6ecfa7',
        '#bfff8c', '#8c54b1', '#773618', '#ffa079', '#a8001f', '#ff1c44', '#5e1123',
        '#679793', '#ff5e93', '#4b6774', '#5291cc', '#aa7031', '#01cffd', '#00c36b',
        '#60345d', '#90d42f', '#bfd47c', '#5044a1', '#4d230c', '#7c5900', '#ffcd44',
        '#8201cf', '#4dfdff', '#89003d', '#7b525b', '#00749c', '#aa8297', '#80708e',
        '#6264fd', '#c33489', '#cd2846', '#ff9ab5', '#c35dba', '#216701', '#008e64',
        '#628023', '#8987bf', '#97ddd4', '#cd7e57', '#d1b65b', '#60006e', '#995444',
        '#afc6db', '#f2ffd1', '#00eb01', '#cd85bc', '#4400c4', '#799c7e', '#727046',
        '#93ffba', '#0054c1', '#ac93eb', '#3fa316', '#5e3a80', '#004b33', '#7cb8d3',
        '#972a00', '#386e64', '#b8005b', '#ff803d', '#ffd1e8', '#802f59', '#213400',
        '#a15d6e', '#4fb5af', '#9e9e46', '#337c3d', '#c14100', '#c6e83d', '#6b05e8',
        '#75bc4f', '#a5c4a8', '#da546e', '#d88e38', '#fb7cff', '#4b6449', '#d6c3eb',
        '#792d36', '#4b8ea5', '#4687ff', '#a300c3', '#e9a3d4', '#ffbc77', '#464800',
        '#a1c6ff', '#90a1e9', '#4f6993', '#e65db1', '#9e90af', '#57502a', '#af5dd4',
        '#856de1', '#c16e72', '#e400e2', '#b8b68a', '#382d00', '#e27ea3', '#ac3b2f',
        '#a8ba4b', '#69b582', '#93d190', '#af8c46', '#075e77', '#009789', '#590f01',
        '#5b7c80', '#2f5726', '#e4643b', '#5e3f28', '#7249bc', '#4b526b', '#c879dd',
        '#9c3190', '#c8e6f2', '#05aaeb', '#a76b9a', '#e6af00', '#60ff62', '#f2dd00',
        '#774401', '#602441', '#677eca', '#799eaf', '#0ce8a1', '#9cf7db', '#830075',
        '#8e6d49', '#e2412f', '#b8496b', '#794985', '#ffcfb5', '#4b5dc6', '#e2b391',
        '#ff4bed', '#d6efa5', '#bf6026', '#d6a3b5', '#bc7c00', '#876eb1', '#ff2fa1',
        '#ffe8af', '#33574b', '#b88c79', '#6b8752', '#bc93d1', '#1ae6fd', '#a13b72',
        '#a350a8', '#6d0097', '#89647b', '#59578a', '#f98e8a', '#e6d67c', '#706b01',
        '#1859ff', '#1626ff', '#00d856', '#f7a1fd', '#79953b', '#b1a7d4', '#7ecfdd',
        '#00caaf', '#79463b', '#daffe6', '#db05b1', '#f2ddff', '#a3e46e', '#891323',
        '#666782', '#e8fd70', '#d8aae8', '#dfbad4', '#fd5269', '#75ae9a', '#9733df',
        '#e4727e', '#8c5926', '#774669', '#2f3da8'
    ]

    # same colors as RGB tuples
    RGB = [ImageColor.getrgb(x) for x in COLORCET_GLASBEY]


if __name__ == "__main__":
    print(SegmentationColors.COLORCET_GLASBEY)
    print(SegmentationColors.RGB)
