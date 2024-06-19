import matplotlib.pyplot as plt
import pickle

def generate_line_graph(data: dict[str, list[tuple[float, float]]], xscale=None, xticks=None, xlabel=None, ylabel=None, filename=None):
    for line_name, points in data.items():
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        plt.plot(x_values, y_values, label=line_name)
    if xscale is not None:
        plt.xscale(xscale)
    if plt.xticks is not None:
        plt.xticks([], minor=True)
        plt.xticks(xticks, labels=xticks)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

# data = {"1-cossim" : {
#     "basic" :           [(1, 0.0150770292), (2, 0.0203312633), (4, 0.0249272095), (8, 0.0302256203), (16, 0.0345196294), (32, 0.0392510953)],
#     "hadamard" :        [(1, 0.0131657504), (2, 0.0153927064), (4, 0.0174454858), (8, 0.0189521735), (16, 0.0208845118), (32, 0.0226017018)],
#     "rot rand" :        [(1, 0.9944813710), (2, 0.9896800704), (4, 0.9838840969), (8, 0.9860888410), (16, 0.9916837822), (32, 0.9897010494)],
#     "rot rand transp" : [(1, 0.6463779916), (2, 0.6970379185), (4, 0.7244749789), (8, 0.7337504278), (16, 0.7515835997), (32, 0.7461530760)],
# }, "test" : {
#     "basic" :           [(1, 0.0150770292), (2, 0.0203312633), (4, 0.0249272095), (8, 0.0302256203), (16, 0.0345196294), (32, 0.0392510953)],
#     "hadamard" :        [(1, 0.0131657504), (2, 0.0153927064), (4, 0.0174454858), (8, 0.0189521735), (16, 0.0208845118), (32, 0.0226017018)],
#     "rot rand" :        [(1, 0.9944813710), (2, 0.9896800704), (4, 0.9838840969), (8, 0.9860888410), (16, 0.9916837822), (32, 0.9897010494)],
#     "rot rand transp" : [(1, 0.6463779916), (2, 0.6970379185), (4, 0.7244749789), (8, 0.7337504278), (16, 0.7515835997), (32, 0.7461530760)],
# }}
# with open("data", "wb") as f:
#     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("data", "rb") as f:
    statistics = pickle.load(f)

stat = "1-cossim"
generate_line_graph(statistics[stat], xscale="log", xticks=[1, 2, 4, 8, 16, 32], xlabel="context length", ylabel=f"{stat} against truth", filename=f"{stat} all.png")
for method, data in statistics[stat].items():
    generate_line_graph({method : data}, xscale="log", xticks=[1, 2, 4, 8, 16, 32], xlabel="context length", ylabel=f"{stat} against truth", filename=f"{stat} {method}.png")