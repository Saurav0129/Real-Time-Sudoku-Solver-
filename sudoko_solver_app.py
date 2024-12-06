import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the pre-trained model
model = keras.models.load_model(r'C:\Users\skbis\Documents\Vs Code\Python\Sudoku Solver\best_model.keras')


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.bitwise_not(threshold)

def main_outline(contours):
    max_area = 0
    biggest = np.array([])  
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reframe(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def splitcells(image):
    rows = np.vsplit(image, 9)
    cells = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for cell in cols:
            cells.append(cell)
    return cells

def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped

def read_cells(cell, model):
    result = []
    for image in cell:
        img = np.asarray(image)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.max(predictions)
        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

# Backtracking Sudoku solver function
def solve(grid):
    row, col = next_box(grid)
    if row == -1 and col == -1:
        return True
    for num in range(1, 10):
        if possible(grid, row, col, num):
            grid[row][col] = num
            if solve(grid):
                return True
            grid[row][col] = 0
    return False

def next_box(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return row, col
    return -1, -1

def possible(grid, row, col, num):
    if num in grid[row]:
        return False
    for r in range(9):
        if grid[r][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if grid[r][c] == num:
                return False
    return True

# Streamlit UI
st.sidebar.title("Navigation")
pages = ["Sudoku Solver", "About"]
selected_page = st.sidebar.selectbox("Go to", pages)

if selected_page == "Sudoku Solver":
    st.title("Sudoku Solver")
    st.write("Upload a Sudoku puzzle image, and the app will solve it for you!")
    
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        puzzle = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        puzzle = cv2.resize(puzzle, (450, 450))

        su_puzzle = preprocess(puzzle)
        su_contour, _ = cv2.findContours(su_puzzle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        su_biggest, _ = main_outline(su_contour)

        if su_biggest.size != 0:
            su_biggest = reframe(su_biggest)
            su_pts1 = np.float32(su_biggest)
            su_pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
            su_matrix = cv2.getPerspectiveTransform(su_pts1, su_pts2)
            su_imagewrap = cv2.warpPerspective(puzzle, su_matrix, (450, 450))
            su_imagewrap = cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)

            sudoku_cell = splitcells(su_imagewrap)
            sudoku_cell_croped = CropCell(sudoku_cell)

            solved_values = read_cells(sudoku_cell_croped, model)
            grid = np.array(solved_values).reshape(9, 9)
            if solve(grid):
                st.write("Solved Sudoku Puzzle:")
                st.table(grid)

                # Download the solved Sudoku as an image
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(grid, cmap="Blues")
                ax.set_xticks(np.arange(9))
                ax.set_yticks(np.arange(9))
                ax.set_xticklabels(np.arange(1, 10))
                ax.set_yticklabels(np.arange(1, 10))
                for i in range(9):
                    for j in range(9):
                        ax.text(j, i, str(grid[i, j]), ha='center', va='center', color='black')
                img_bytes = fig_to_png(fig)
                st.download_button("Download Solved Sudoku Image", img_bytes, file_name="solved_sudoku.png", mime="image/png")
                
            else:
                st.write("No solution exists for the Sudoku puzzle.")

elif selected_page == "About":
    st.title("About Sudoku Solver App")
    st.write("""
    This is a Sudoku Solver application built using deep learning. The app uses a pre-trained model to detect and solve a Sudoku puzzle from an image. 
    It employs OpenCV for image preprocessing, contour detection, and perspective transformation to extract the puzzle grid. 
    The app utilizes a TensorFlow model to recognize the digits in the grid and uses a backtracking algorithm to solve it.

    ### Features:
    - Upload an image of a Sudoku puzzle.
    - Automatically solve the puzzle with a deep learning model.
    - Download the solved puzzle image.""")
    
    st.markdown("### GitHub Repository")
    github_link = "https://github.com/Saurav0129/Real-Time-Sudoku-Solver-"
    st.markdown(
        f"Check out the [GitHub repository]({github_link}) for more details and to access the source code."
    )
