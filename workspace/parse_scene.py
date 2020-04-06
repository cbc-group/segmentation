import h5py
import io


def main():
    with h5py.File("fused_scene.ims", "r") as h:
        byte_str = bytes(h["Scene/Data"][0])
        text = byte_str.decode("UTF-8")

        with open('scene.xml', 'w') as f:
            f.write(text)


if __name__ == "__main__":
    main()
