# Details for Running Quadruped
This readme provides some notes on running the quadruped environment "as is" so far. 
- This environment is from the [walk in the park](https://github.com/ikostrikov/walk_in_the_park/tree/main) project
- The versioning (as of writing) is 3+ years old some package's it uses are out of date
- Running the environment doesn't depend on doing their full list of dependencies. If you set-up libero & crossformer before, most of them are handled via that. 

## Known potential issues:
- it uses a specific copy of [dmcgym](https://github.com/ikostrikov/dmcgym):
```
pip install https://github.com/ikostrikov/dmcgym
```
- the above package uses mujoco 2.2.0 which is too old for robosuite 1.5.1 (which uses 3.1+ mujoco), so we need to see if we can upgrade it
- the version of "gym" it depends on is apparently bugged (that's the warning recieved on set-up) 


