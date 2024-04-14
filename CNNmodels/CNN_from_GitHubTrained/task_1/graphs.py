from tensorboard import program
import webbrowser

log_dir = './task_1_my_run/'
# log_dir = './task_1_new_run'     # new run log directory
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"Tensorflow listening on {url}")
webbrowser.open_new('http://localhost:6006/')
 
# Kill process
# Windows
# netstat -ano | findstr :6006
# taskkill /F /PID {PID}