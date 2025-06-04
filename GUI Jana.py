import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import cv2
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import random
import math
import time

Disease_name = []
Disease_symptoms = []
Disease_treatment = []

class MedicalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical Project")
        self.geometry("1000x600")
        self.G = nx.Graph()
        self.splash_screen()
        self.main_page()

    def splash_screen(self):
        splash_frame = tk.Frame(self)
        splash_frame.pack(fill=tk.BOTH, expand=True)

        # Load video
        video = cv2.VideoCapture('startup video.mp4')  # Update the file path with your video file
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        canvas = tk.Canvas(splash_frame, width=frame_width, height=frame_height)
        canvas.pack()

        ret, frame = video.read()
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(frame, "Welcome to the Medical Scraper", (80, 300), cv2.QT_FONT_BLACK, 2, (255, 255, 255), 2, cv2.LINE_AA)

            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.update()
            time.sleep(0.25 / fps)
            ret, frame = video.read()

        splash_frame.destroy()

    def main_page(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.background_image = Image.open("background.jpg")  # Update with your image file path
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        bg_label = tk.Label(main_frame, image=self.background_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        tab_control = ttk.Notebook(main_frame)
        self.web_scraping_tab = ttk.Frame(tab_control)
        self.network_construction_tab = ttk.Frame(tab_control)
        self.network_analysis_tab = ttk.Frame(tab_control)
        self.heatmap_tab = ttk.Frame(tab_control)
        self.three_d_model_tab = ttk.Frame(tab_control)
        self.bonus_tab = ttk.Frame(tab_control)  # Add Bonus tab

        tab_control.add(self.web_scraping_tab, text='Web Scraping')
        tab_control.add(self.network_construction_tab, text='Network Construction')
        tab_control.add(self.network_analysis_tab, text='Network Analysis')
        tab_control.add(self.heatmap_tab, text='Heatmap')
        tab_control.add(self.three_d_model_tab, text='3D Model')
        tab_control.add(self.bonus_tab, text='Graphs')  # Add Bonus tab to tab control

        tab_control.pack(expand=1, fill='both')

        self.setup_web_scraping_tab(self.web_scraping_tab)
        self.setup_network_construction_tab(self.network_construction_tab)
        self.setup_network_analysis_tab(self.network_analysis_tab)
        self.setup_heatmap_tab(self.heatmap_tab)
        self.setup_3d_model_tab(self.three_d_model_tab)
        self.setup_bonus_tab(self.bonus_tab)  # Setup the new Bonus tab

    def setup_web_scraping_tab(self, tab):
        background_image = Image.open("AI-web-scraping-.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        bg_label = tk.Label(self.web_scraping_tab, image=background_photo)
        bg_label.image = background_photo  # keep a reference!
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(tab, text="Extract and display disease data from web sources.", font=("Helvetica", 16))
        label.pack(pady=10)
        show_data_button = tk.Button(tab, text="Show Data", command=self.show_web_scraping_data)
        show_data_button.pack(pady=10)
        self.data_text = tk.Text(tab, height=80, width=120)
        self.data_text.pack(pady=10)

    def setup_network_construction_tab(self, tab):
        background_image = Image.open("network.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        bg_label = tk.Label(self.network_construction_tab, image=background_photo)
        bg_label.image = background_photo  # keep a reference!
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(tab, text="Visualize the relationship between diseases based on symptoms and treatments.", font=("Helvetica", 16))
        label.pack(pady=10)
        show_network_button = tk.Button(tab, text="Show Network", command=lambda: self.show_network(self.show_web_scraping_data()))
        show_network_button.pack(pady=10)

    def setup_network_analysis_tab(self, tab):
        background_image = Image.open("network analysis.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        bg_label = tk.Label(self.network_analysis_tab, image=background_photo)
        bg_label.image = background_photo  # keep a reference!
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(tab, text="Analyze the network to find key diseases and their connections.", font=("Helvetica", 16))
        label.pack(pady=10)
        analysis_button = tk.Button(tab, text="Analyze Network", command=self.show_network_analysis)
        analysis_button.pack(pady=10)
        self.analysis_text = tk.Text(tab, height=80, width=80)
        self.analysis_text.pack(pady=10)

    def setup_heatmap_tab(self, tab):
        background_image = Image.open("heatmap.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        bg_label = tk.Label(self.heatmap_tab, image=background_photo)
        bg_label.image = background_photo  # keep a reference!
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(tab, text="Show a heatmap of disease occurrences.", font=("Helvetica", 16))
        label.pack(pady=10)
        show_heatmap_button = tk.Button(tab, text="Show Heatmap", command=self.show_heatmap)
        show_heatmap_button.pack(pady=10)

    def setup_3d_model_tab(self, tab):
        background_image = Image.open("3d model.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        bg_label = tk.Label(self.three_d_model_tab, image=background_photo)
        bg_label.image = background_photo  # keep a reference!
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        label = tk.Label(tab, text="3D visualization of the disease relationships network.", font=("Helvetica", 16))
        label.pack(pady=10)
        show_3d_model_button = tk.Button(tab, text="Show 3D Model", command=self.show_3d_model)
        show_3d_model_button.pack(pady=10)

    def setup_bonus_tab(self, tab):

        label = tk.Label(tab, text="Bonus part: Show additional graphs.", font=("Helvetica", 16))
        label.pack(pady=10)
        show_graphs_button = tk.Button(tab, text="Show Graphs", command=self.show_graphs)
        show_graphs_button.pack(pady=10)

    def show_web_scraping_data(self):
        api_key = '9cbf310e010773d8f8a74b03c78cb1857c5907ad56b98431d9b40e8302e37787'
        query = 'site:huggingface.co Diseases symptoms datasets'
        search = GoogleSearch({"q": query, "api_key": api_key})
        results = search.get_dict()
        datasets = []
        for result in results['organic_results']:
            dataset_name = result['title']
            dataset_url = result['link']
            datasets.append({'name': dataset_name, 'url': dataset_url})

        dataset_index = 3
        mylink = datasets[dataset_index]['url']

        r = requests.get(mylink)
        x = BeautifulSoup(r.content, "html.parser")
        div_tag = x.find_all('div', class_='line-clamp-2')

        All_text = []
        for i in range(len(div_tag)):
            text = div_tag[i].text.strip()
            All_text.append(text)

        nums = []
        for i in range(0, len(All_text), 4):
            nums.append(All_text[i])

        for i in range(1, len(All_text), 4):
            Disease_name.append(All_text[i])

        for i in range(2, len(All_text), 4):
            Disease_symptoms.append(All_text[i])

        for i in range(3, len(All_text), 4):
            Disease_treatment.append(All_text[i])

        self.data_text.delete('1.0', tk.END)  # Clear previous data
        self.data_text.insert(tk.END, "All the search results: \n\n")
        for dataset in datasets:
            self.data_text.insert(tk.END, str(dataset['name']) +"\t"+str( dataset['url'])+"\n")
        self.data_text.insert(tk.END, "\nMy dataset is: "+str(mylink)+"\n\n")
        self.data_text.insert(tk.END, "Name: {}\n\n\n\n".format(Disease_name))
        self.data_text.insert(tk.END, "Symptoms: {}\n\n\n\n".format(Disease_symptoms))
        self.data_text.insert(tk.END, "Treatment: {}\n\n\n\n".format(Disease_treatment))
        return Disease_name, Disease_symptoms, Disease_treatment

    def show_network(self, Disease_information):
        nodes = []
        splitted_symptoms = []
        for i in range(len((Disease_information[0]))):
            splitted_symptom = (Disease_information[1])[i].split(",")
            splitted_symptoms.append(splitted_symptom)
        splitted_treatments = []
        for i in range(len((Disease_information[0]))):
            splitted_treatment = (Disease_information[2])[i].split(",")
            splitted_treatments.append(splitted_treatment)
        for i in range(len((Disease_information[0]))):
            for j in range(i+1, len(Disease_information[0])):
                for symptom in splitted_symptoms[j]:
                    if symptom in splitted_symptoms[i]:
                        nodes.append((Disease_information[0])[i])
                        nodes.append((Disease_information[0])[j])
                        self.G.add_edge(Disease_name[i], (Disease_information[0])[j])
        for i in range(len((Disease_information[0]))):
            for j in range(i+1, len((Disease_information[0]))):
                for symptom in splitted_treatments[j]:
                    if symptom in splitted_treatments[i]:
                        nodes.append((Disease_information[0])[i])
                        nodes.append((Disease_information[0])[j])
                        self.G.add_edge((Disease_information[0])[i], (Disease_information[0])[j])
        self.G.add_nodes_from(nodes)
        pos = nx.shell_layout(self.G)
        plt.figure(figsize=(10,10))
        nx.draw(self.G, pos, with_labels=True, node_size=300)
        plt.show()
        return self.G

    def show_network_analysis(self):
        self.analysis_text.delete('1.0', tk.END)
        nodes_degree = {node: self.G.degree[node] for node in self.G.nodes}
        self.analysis_text.insert(tk.END, "Nodes degree:\n")
        for node, degree in nodes_degree.items():
            self.analysis_text.insert(tk.END, f"Node {node}: >>>>>>> {degree}\n")

        false_betweenness = nx.betweenness_centrality(self.G, normalized=False)
        true_betweenness = nx.betweenness_centrality(self.G)

        self.analysis_text.insert(tk.END, "\nBetweenness with false normalization:\n")
        self.analysis_text.insert(tk.END, f"{false_betweenness}\n")

        self.analysis_text.insert(tk.END, "\nBetweenness with true normalization:\n")
        self.analysis_text.insert(tk.END, f"{true_betweenness}\n")

        max_false = max(false_betweenness.values(), default=0)
        max_true = max(true_betweenness.values(), default=0)
        max_true_node = max(true_betweenness, default='N/A')

        self.analysis_text.insert(tk.END, f"Max False: {max_false}\n")
        self.analysis_text.insert(tk.END, f"Max True: {max_true}\n")
        self.analysis_text.insert(tk.END, f"The most central disease (in symptoms and treatment) is {max_true_node}\n")


    def show_3d_model(self):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        pos = nx.spring_layout(self.G, dim=3)
        for node, (x, y, z) in pos.items():
            ax.scatter(x, y, z, label=node, s=20)
        for edge in self.G.edges():
            ax.plot([pos[edge[0]][0], pos[edge[1]][0]],
                    [pos[edge[0]][1], pos[edge[1]][1]],
                    [pos[edge[0]][2], pos[edge[1]][2]])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("The 3D model of the network")
        plt.show()

    def show_heatmap(self):
        xs = []
        ys = []
        for i in range(100):
            x = random.randint(630, 650)
            xs.append(x)

            y = random.randint(780, 800)
            ys.append(y)


        for i in range(len(xs)):
            plt.plot(xs[i], ys[i], "ro")
        plt.show()

        grid_size = 1
        h = 10

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_grid = np.arange(x_min - h, x_max + h, grid_size)
        y_grid = np.arange(y_min - h, y_max + h, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        xc = x_mesh + (grid_size / 2)
        yc = y_mesh + (grid_size / 2)

        def kde_quartic(d, h):
            dn = d / h
            P = (15 / 16) * (1 - dn ** 2) * 2
            return P

        intensity_list = []
        for r in range(len(xc)):
            intensity_row = []
            for c in range(len(xc[0])):
                kde_value_list = []
                for i in range(len(xs)):
                    d = math.sqrt((xc[r][c] - xs[i]) ** 2 + (yc[r][c] - ys[i]) ** 2)
                    if d <= h:
                        p = kde_quartic(d, h)
                    else:
                        p = 0
                    kde_value_list.append(p)
                p_total = sum(kde_value_list)
                intensity_row.append(p_total)
            intensity_list.append(intensity_row)
        intensity = np.array(intensity_list)
        plt.pcolormesh(x_mesh, y_mesh, intensity)
        plt.plot(xs, ys, 'ro')
        fig1 = plt.gcf()
        plt.axis("off")
        plt.show()
        plt.draw()
        fig1.savefig("hmap.jpg")
        S1 = cv2.imread('hmap.jpg')
        S2 = cv2.imread('Human_body_silhouette.png')
        S2 = cv2.cvtColor(S2, cv2.COLOR_BGR2RGB)
        # plt.imshow(S2)
        S22 = cv2.resize(S1, (S2.shape[1], S2.shape[0]), interpolation=cv2.INTER_AREA)
        alpha = 0.5
        image_new = cv2.addWeighted(S22, alpha, S2, 1 - alpha, 0)
        plt.imshow(image_new)
        # plt.imshow(S1)
        plt.axis("off")
        plt.show()

    def show_graphs(self):
        Diseases = self.show_web_scraping_data()
        symptoms_count = [len(symptoms.split(',')) for symptoms in Diseases[1]]

        # Bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(Diseases[0], symptoms_count, color='blue')
        plt.xlabel('Diseases')
        plt.ylabel('Number of Symptoms')
        plt.title('Number of Symptoms for Each Disease')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(symptoms_count, bins=range(1, max(symptoms_count)+1), color='green', alpha=0.7)
        plt.xlabel('Number of Symptoms')
        plt.ylabel('Number of Diseases')
        plt.title('Distribution of Number of Symptoms Across Diseases')
        plt.grid(axis='y')
        plt.show()

if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()
